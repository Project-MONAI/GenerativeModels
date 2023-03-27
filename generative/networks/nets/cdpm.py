# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import abstractmethod

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from generative.networks.layers.RPE import RPEAttention
from monai.networks.blocks import Convolution


class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """


class TimestepEmbedAttnThingsSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes extra things to the children that
    support it as an extra input.
    """
    def forward(self, x, emb, attn_mask, T=1, frame_indices=None)-> torch.Tensor:
        for layer in self:
            if isinstance(layer, TimestepBlock):
                kwargs = dict(emb=emb)
                kwargs['emb'] = emb
            elif isinstance(layer, FactorizedAttentionBlock):
                kwargs = dict(
                    temb=emb,
                    attn_mask=attn_mask,
                    T=T,
                    frame_indices=frame_indices,
                )
            else:
                kwargs = {}
            x = layer(x, **kwargs)
        return x


class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2)-> None:
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = Convolution(dims, channels, channels,  padding=1)

    def forward(self, x)-> torch.Tensor:
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.
    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2)-> None:
        super().__init__()
        self.channels = channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = Convolution(dims, channels, channels, strides=stride, padding=1)
        else:
            self.op = avg_pool_nd(stride)

    def forward(self, x)-> torch.Tensor:
        assert x.shape[1] == self.channels
        return self.op(x)


class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
    )-> None:
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv

        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            Convolution(dims, channels, self.out_channels, padding=1),
        )
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(32, self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            Convolution(dims, self.out_channels, self.out_channels, padding=1),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = Convolution(
                dims, channels, self.out_channels, padding=1
            )
        else:
            self.skip_connection = Convolution(dims, channels, self.out_channels, kernel_size = 1)

    def forward(self, x, emb)-> torch.Tensor:
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h


class FactorizedAttentionBlock(nn.Module):

    def __init__(self, channels, num_heads, time_embed_dim=None)-> None:
        super().__init__()
        self.spatial_attention = RPEAttention(
            channels=channels, num_heads=num_heads, time_embed_dim = time_embed_dim, use_rpe_q=False, use_rpe_k=False, use_rpe_v=False,
        )
        self.temporal_attention = RPEAttention(
            channels=channels, num_heads=num_heads,
            time_embed_dim=time_embed_dim,
        )

    def forward(self, x, attn_mask, temb, T, frame_indices=None)-> torch.Tensor:
        BT, C, H, W = x.shape
        B = BT//T
        # reshape to have T in the last dimension becuase that's what we attend over
        x = x.view(B, T, C, H, W).permute(0, 3, 4, 2, 1)  # B, H, W, C, T
        x = x.reshape(B, H*W, C, T)
        x = self.temporal_attention(x,
                                    temb,
                                    frame_indices,
                                    attn_mask=attn_mask.flatten(start_dim=2).squeeze(dim=2), # B x T
                                    )

        # Now we attend over the spatial dimensions by reshaping the input
        x = x.view(B, H, W, C, T).permute(0, 4, 3, 1, 2)  # B, T, C, H, W
        x = x.reshape(B, T, C, H*W)
        x = self.spatial_attention(x,
                                   temb,
                                   frame_indices=None,
                                   )
        x = x.reshape(BT, C, H, W)
        return x


class UNet_2Plus1_Model(nn.Module):
    """
    The full UNet model with attention and timestep embedding.
    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_heads: the number of attention heads in each attention layer.
    """

    def __init__(
        self,
        in_channels,
        model_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        image_size=None,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_heads=1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
    )-> None:
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.in_channels = in_channels + 1
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_heads = num_heads
        self.num_heads_upsample = num_heads_upsample

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedAttnThingsSequential(
                    Convolution(dims, self.in_channels, model_channels, padding=1)
                )
            ]
        )
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        # channel_mult=(1, 2, 4, 8),
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch, num_heads=num_heads, time_embed_dim=time_embed_dim,
                        )
                    )
                self.input_blocks.append(TimestepEmbedAttnThingsSequential(*layers))
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                self.input_blocks.append(
                    TimestepEmbedAttnThingsSequential(Downsample(ch, conv_resample, dims=dims))
                )
                input_block_chans.append(ch)
                ds *= 2

        self.middle_block = TimestepEmbedAttnThingsSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            FactorizedAttentionBlock(ch, num_heads=num_heads, time_embed_dim=time_embed_dim),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = [
                    ResBlock(
                        ch + input_block_chans.pop(),
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        FactorizedAttentionBlock(
                            ch,
                            num_heads=num_heads_upsample,
                            time_embed_dim=time_embed_dim,
                        )
                    )
                if level and i == num_res_blocks:
                    layers.append(Upsample(ch, conv_resample, dims=dims))
                    ds //= 2
                self.output_blocks.append(TimestepEmbedAttnThingsSequential(*layers))

        self.out = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            Convolution(dims, model_channels, out_channels, padding=1),
        )

    @property
    def inner_dtype(self):
        """
        Get the dtype used by the torso of the model.
        """
        return next(self.input_blocks.parameters()).dtype

    def forward(self, x, timesteps, context=None)-> torch.Tensor:
        """
        Apply the model to an input batch.
        :param x: list of both an [N x C x ...] Tensor of inputs, noisy input, and target.
        :param x0: an [N x C x ...] Tensor of inputs, target.
        :param timesteps: a 1-D batch of timesteps.
        :param context: all other informations, including:
            :param frame_indices: absolute index in the whole volume.
            :param obs_mask: absolute index in the whole volume.
            :param latent_mask: absolute index in the whole volume.

        :return: an [N x C x ...] Tensor of outputs.
        """
        frame_indices = context[0]
        obs_mask = context[1]
        latent_mask = context[2]
        x0 = context[3]

        B, T, C, H, W = x.shape
        timesteps = timesteps.view(B, 1).expand(B, T)
        attn_mask = (obs_mask + latent_mask).clip(max=1)
        # add channel to indicate obs
        indicator_template = torch.ones_like(x[:, :, :1, :, :])
        obs_indicator = indicator_template * obs_mask
        x = torch.cat([x*(1-obs_mask) + x0*obs_mask,
                    obs_indicator],
                   dim=2,
        )
        x = x.reshape(B*T, self.in_channels, H, W)
        timesteps = timesteps.reshape(B*T)
        hs = []
        emb = self.timestep_embedding(timesteps, self.model_channels*4)
        # print(f't_emb.shape is {t_emb.shape}')
        # emb = self.time_embed(t_emb)
        h = x.type(self.inner_dtype)

        for layer, module in enumerate(self.input_blocks):
            h = module(h, emb,  attn_mask, T=T, frame_indices=frame_indices)
            hs.append(h)
        h = self.middle_block(h, emb,  attn_mask, T=T, frame_indices=frame_indices)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb,  attn_mask, T=T, frame_indices=frame_indices)
        h = h.type(x.dtype)
        out = self.out(h)
        return out.view(B, T, self.out_channels, H, W)

    def timestep_embedding(self, timesteps, dim, max_period=10000)-> torch.Tensor:
        """
        Create sinusoidal timestep embeddings.
        :param timesteps: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an [N x dim] Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=timesteps.device)
        args = timesteps[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def get_feature_vectors(self, x, timesteps, y=None)-> torch.Tensor:
        """
        Apply the model and return all of the intermediate tensors.
        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: a dict with the following keys:
                 - 'down': a list of hidden state tensors from downsampling.
                 - 'middle': the tensor of the output of the lowest-resolution
                             block in the model.
                 - 'up': a list of hidden state tensors from upsampling.
        """
        hs = []
        emb = self.timestep_embedding(timesteps, self.model_channels)
        # emb = self.time_embed(t_emb)
        result = dict(down=[], up=[])
        h = x.type(self.inner_dtype)
        for module in self.input_blocks:
            h = module(h, emb)
            hs.append(h)
            result["down"].append(h.type(x.dtype))
        h = self.middle_block(h, emb)
        result["middle"] = h.type(x.dtype)
        for module in self.output_blocks:
            cat_in = torch.cat([h, hs.pop()], dim=1)
            h = module(cat_in, emb)
            result["up"].append(h.type(x.dtype))
        return result

    def sample_some_indices(self, max_indices=16, T=32):
        s = torch.randint(low=1, high=max_indices+1, size=())
        max_scale = T / (s-0.999)
        scale = np.exp(np.random.rand() * np.log(max_scale))
        pos = torch.rand(()) * (T - scale*(s-1))
        indices = [int(pos+i*scale) for i in range(s)]
        # do some recursion if we have somehow failed to satisfy the consrtaints
        if all(i<T and i>=0 for i in indices):
            return indices
        else:
            print('warning: sampled invalid indices', [int(pos+i*scale) for i in range(s)], 'trying again')
            # exit()
            return self.sample_some_indices(max_indices, T)

    def get_image_context(self, batch1, batch2=None, max_frames=16, set_masks={'obs': (), 'latent': ()}):
        N = max_frames

        B, T, *_ = batch1.shape
        masks = {k: torch.zeros_like(batch1[:, :, :1, :1, :1]) for k in ['obs', 'latent']}
        for obs_row, latent_row in zip(*[masks[k] for k in ['obs', 'latent']]):
            latent_row[self.sample_some_indices(max_indices=N, T=T)] = 1.
            while True:
                mask = obs_row if torch.rand(()) < 0.5 else latent_row
                indices = torch.tensor(self.sample_some_indices(max_indices=N, T=T))
                taken = (obs_row[indices] + latent_row[indices]).view(-1)
                indices = indices[taken == 0]  # remove indices that are already used in a mask
                if len(indices) > N - sum(obs_row) - sum(latent_row):
                    break
                mask[indices] = 1.
        if len(set_masks['obs']) > 0:  # set_masks allow us to choose informative masks for logging
            for k in masks:
                set_values = set_masks[k]
                n_set = min(len(set_values), len(masks[k]))
                masks[k][:n_set] = set_values[:n_set]
        any_mask = (masks['obs'] + masks['latent']).clip(max=1)

        batch, (obs_mask, latent_mask), frame_indices =\
            self.prepare_training_batch(
                any_mask, batch1, batch2, (masks['obs'], masks['latent']), max_frames
            )
        return (frame_indices, obs_mask, latent_mask, batch)

    def prepare_training_batch(self, mask, batch1, batch2, tensors, max_frames):
        """
        Prepare training batch by selecting frames from batch1 according to mask, appending uniformly sampled frames
        from batch2, and selecting the corresponding elements from tensors (usually obs_mask and latent_mask).
        """
        B, T, *_ = mask.shape
        mask = mask.view(B, T)  # remove unit C, H, W dims
        effective_T = max_frames
        indices = torch.zeros_like(mask[:, :effective_T], dtype=torch.int64)
        new_batch = torch.zeros_like(batch1[:, :effective_T])
        new_tensors = [torch.zeros_like(t[:, :effective_T]) for t in tensors]
        for b in range(B):
            instance_T = mask[b].sum().int()
            indices[b, :instance_T] = mask[b].nonzero().flatten()
            indices[b, instance_T:] = torch.randint_like(indices[b, instance_T:], high=T)
            new_batch[b, :instance_T] = batch1[b][mask[b]==1]
            new_batch[b, instance_T:] = (batch1 if batch2 is None else batch2)[b][indices[b, instance_T:]]
            for new_t, t in zip(new_tensors, tensors):
                new_t[b, :instance_T] = t[b][mask[b]==1]
                new_t[b, instance_T:] = t[b][indices[b, instance_T:]]
        return new_batch, new_tensors, indices

    def next_indices(self, done_frames, images_length, max_frames = 16, step_size = None):
        if step_size is None:
            step_size = max_frames//2
        if len(done_frames) == 1:
            obs_frame_indices = [0]
            latent_frame_indices = list(range(1, max_frames))
        else:
            obs_frame_indices = sorted(done_frames)[-(max_frames - step_size):]
            first_idx = obs_frame_indices[-1] + 1
            latent_frame_indices = list(range(first_idx, min(first_idx + step_size, images_length)))

        obs_mask = torch.cat([torch.ones_like(torch.tensor(obs_frame_indices)),
                              torch.zeros_like(torch.tensor(latent_frame_indices))]).view(1, -1, 1, 1, 1).float()
        latent_mask = 1 - obs_mask

        return obs_frame_indices, latent_frame_indices, obs_mask, latent_mask

if __name__ == "__main__":
    Model = UNet_2Plus1_Model(
        in_channels = 1,
        out_channels= 1,
        model_channels = 128,
        num_res_blocks=2,
        attention_resolutions=[16, 8],
        )
    input = torch.randn((1,20,1,128,128))
    timesteps = torch.randn((1,1))
    pred = Model(input, x0=input , timesteps=timesteps)
    print(f'pred is with shape {pred.shape}')
