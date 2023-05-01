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
#
# =========================================================================
# Adapted from https://github.com/huggingface/diffusers
# which has the following license:
# https://github.com/huggingface/diffusers/blob/main/LICENSE
#
# Copyright 2022 UC Berkeley Team and The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =========================================================================

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from monai.networks.blocks import Convolution
from monai.utils import ensure_tuple_rep
from torch import nn

from generative.networks.nets.diffusion_model_unet import get_down_block, get_mid_block, get_timestep_embedding


class ControlNetConditioningEmbedding(nn.Module):
    """
    Network to encode the conditioning into a latent space.
    """

    def __init__(
        self, spatial_dims: int, in_channels: int, out_channels: int, num_channels: Sequence[int] = (16, 32, 96, 256)
    ):
        super().__init__()

        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        self.blocks = nn.ModuleList([])

        for i in range(len(num_channels) - 1):
            channel_in = num_channels[i]
            channel_out = num_channels[i + 1]
            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_in,
                    strides=1,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )

            self.blocks.append(
                Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=channel_in,
                    out_channels=channel_out,
                    strides=2,
                    kernel_size=3,
                    padding=1,
                    conv_only=True,
                )
            )

        self.conv_out = zero_module(
            Convolution(
                spatial_dims=spatial_dims,
                in_channels=num_channels[-1],
                out_channels=out_channels,
                strides=1,
                kernel_size=3,
                padding=1,
                conv_only=True,
            )
        )

    def forward(self, conditioning):
        embedding = self.conv_in(conditioning)
        embedding = F.silu(embedding)

        for block in self.blocks:
            embedding = block(embedding)
            embedding = F.silu(embedding)

        embedding = self.conv_out(embedding)

        return embedding


def zero_module(module):
    for p in module.parameters():
        nn.init.zeros_(p)
    return module


class ControlNet(nn.Module):
    """
    Control network for diffusion models based on Zhang and Agrawala "Adding Conditional Control to Text-to-Image
    Diffusion Models" (https://arxiv.org/abs/2302.05543)

    Args:
        spatial_dims: number of spatial dimensions.
        in_channels: number of input channels.
        num_res_blocks: number of residual blocks (see ResnetBlock) per level.
        num_channels: tuple of block output channels.
        attention_levels: list of levels to add attention.
        norm_num_groups: number of groups for the normalization.
        norm_eps: epsilon for the normalization.
        resblock_updown: if True use residual blocks for up/downsampling.
        num_head_channels: number of channels in each attention head.
        with_conditioning: if True add spatial transformers to perform conditioning.
        transformer_num_layers: number of layers of Transformer blocks to use.
        cross_attention_dim: number of context dimensions to use.
        num_class_embeds: if specified (as an int), then this model will be class-conditional with `num_class_embeds`
            classes.
        upcast_attention: if True, upcast attention operations to full precision.
        use_flash_attention: if True, use flash attention for a memory efficient attention mechanism.
        conditioning_embedding_in_channels: number of input channels for the conditioning embedding.
        conditioning_embedding_num_channels: number of channels for the blocks in the conditioning embedding.
    """

    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        num_res_blocks: Sequence[int] | int = (2, 2, 2, 2),
        num_channels: Sequence[int] = (32, 64, 64, 64),
        attention_levels: Sequence[bool] = (False, False, True, True),
        norm_num_groups: int = 32,
        norm_eps: float = 1e-6,
        resblock_updown: bool = False,
        num_head_channels: int | Sequence[int] = 8,
        with_conditioning: bool = False,
        transformer_num_layers: int = 1,
        cross_attention_dim: int | None = None,
        num_class_embeds: int | None = None,
        upcast_attention: bool = False,
        use_flash_attention: bool = False,
        conditioning_embedding_in_channels: int = 1,
        conditioning_embedding_num_channels: Sequence[int] | None = (16, 32, 96, 256),
    ) -> None:
        super().__init__()
        if with_conditioning is True and cross_attention_dim is None:
            raise ValueError(
                "DiffusionModelUNet expects dimension of the cross-attention conditioning (cross_attention_dim) "
                "when using with_conditioning."
            )
        if cross_attention_dim is not None and with_conditioning is False:
            raise ValueError(
                "DiffusionModelUNet expects with_conditioning=True when specifying the cross_attention_dim."
            )

        # All number of channels should be multiple of num_groups
        if any((out_channel % norm_num_groups) != 0 for out_channel in num_channels):
            raise ValueError("DiffusionModelUNet expects all num_channels being multiple of norm_num_groups")

        if len(num_channels) != len(attention_levels):
            raise ValueError("DiffusionModelUNet expects num_channels being same size of attention_levels")

        if isinstance(num_head_channels, int):
            num_head_channels = ensure_tuple_rep(num_head_channels, len(attention_levels))

        if len(num_head_channels) != len(attention_levels):
            raise ValueError(
                "num_head_channels should have the same length as attention_levels. For the i levels without attention,"
                " i.e. `attention_level[i]=False`, the num_head_channels[i] will be ignored."
            )

        if isinstance(num_res_blocks, int):
            num_res_blocks = ensure_tuple_rep(num_res_blocks, len(num_channels))

        if len(num_res_blocks) != len(num_channels):
            raise ValueError(
                "`num_res_blocks` should be a single integer or a tuple of integers with the same length as "
                "`num_channels`."
            )

        if use_flash_attention is True and not torch.cuda.is_available():
            raise ValueError(
                "torch.cuda.is_available() should be True but is False. Flash attention is only available for GPU."
            )

        self.in_channels = in_channels
        self.block_out_channels = num_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.num_head_channels = num_head_channels
        self.with_conditioning = with_conditioning

        # input
        self.conv_in = Convolution(
            spatial_dims=spatial_dims,
            in_channels=in_channels,
            out_channels=num_channels[0],
            strides=1,
            kernel_size=3,
            padding=1,
            conv_only=True,
        )

        # time
        time_embed_dim = num_channels[0] * 4
        self.time_embed = nn.Sequential(
            nn.Linear(num_channels[0], time_embed_dim), nn.SiLU(), nn.Linear(time_embed_dim, time_embed_dim)
        )

        # class embedding
        self.num_class_embeds = num_class_embeds
        if num_class_embeds is not None:
            self.class_embedding = nn.Embedding(num_class_embeds, time_embed_dim)

        # control net conditioning embedding
        self.controlnet_cond_embedding = ControlNetConditioningEmbedding(
            spatial_dims=spatial_dims,
            in_channels=conditioning_embedding_in_channels,
            num_channels=conditioning_embedding_num_channels,
            out_channels=num_channels[0],
        )

        # down
        self.down_blocks = nn.ModuleList([])
        self.controlnet_down_blocks = nn.ModuleList([])
        output_channel = num_channels[0]

        controlnet_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=output_channel,
            out_channels=output_channel,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        controlnet_block = zero_module(controlnet_block.conv)
        self.controlnet_down_blocks.append(controlnet_block)

        for i in range(len(num_channels)):
            input_channel = output_channel
            output_channel = num_channels[i]
            is_final_block = i == len(num_channels) - 1

            down_block = get_down_block(
                spatial_dims=spatial_dims,
                in_channels=input_channel,
                out_channels=output_channel,
                temb_channels=time_embed_dim,
                num_res_blocks=num_res_blocks[i],
                norm_num_groups=norm_num_groups,
                norm_eps=norm_eps,
                add_downsample=not is_final_block,
                resblock_updown=resblock_updown,
                with_attn=(attention_levels[i] and not with_conditioning),
                with_cross_attn=(attention_levels[i] and with_conditioning),
                num_head_channels=num_head_channels[i],
                transformer_num_layers=transformer_num_layers,
                cross_attention_dim=cross_attention_dim,
                upcast_attention=upcast_attention,
                use_flash_attention=use_flash_attention,
            )

            self.down_blocks.append(down_block)

            for _ in range(num_res_blocks[i]):
                controlnet_block = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=output_channel,
                    out_channels=output_channel,
                    strides=1,
                    kernel_size=1,
                    padding=0,
                    conv_only=True,
                )
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)
            #
            if not is_final_block:
                controlnet_block = Convolution(
                    spatial_dims=spatial_dims,
                    in_channels=output_channel,
                    out_channels=output_channel,
                    strides=1,
                    kernel_size=1,
                    padding=0,
                    conv_only=True,
                )
                controlnet_block = zero_module(controlnet_block)
                self.controlnet_down_blocks.append(controlnet_block)

        # mid
        mid_block_channel = num_channels[-1]

        self.middle_block = get_mid_block(
            spatial_dims=spatial_dims,
            in_channels=mid_block_channel,
            temb_channels=time_embed_dim,
            norm_num_groups=norm_num_groups,
            norm_eps=norm_eps,
            with_conditioning=with_conditioning,
            num_head_channels=num_head_channels[-1],
            transformer_num_layers=transformer_num_layers,
            cross_attention_dim=cross_attention_dim,
            upcast_attention=upcast_attention,
            use_flash_attention=use_flash_attention,
        )

        controlnet_block = Convolution(
            spatial_dims=spatial_dims,
            in_channels=output_channel,
            out_channels=output_channel,
            strides=1,
            kernel_size=1,
            padding=0,
            conv_only=True,
        )
        controlnet_block = zero_module(controlnet_block)
        self.controlnet_mid_block = controlnet_block

    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        controlnet_cond: torch.Tensor,
        conditioning_scale: float = 1.0,
        context: torch.Tensor | None = None,
        class_labels: torch.Tensor | None = None,
    ) -> tuple[tuple[torch.Tensor], torch.Tensor]:
        """
        Args:
            x: input tensor (N, C, SpatialDims).
            timesteps: timestep tensor (N,).
            controlnet_cond: controlnet conditioning tensor (N, C, SpatialDims).
            conditioning_scale: conditioning scale.
            context: context tensor (N, 1, ContextDim).
            class_labels: context tensor (N, ).
        """
        # 1. time
        t_emb = get_timestep_embedding(timesteps, self.block_out_channels[0])

        # timesteps does not contain any weights and will always return f32 tensors
        # but time_embedding might actually be running in fp16. so we need to cast here.
        # there might be better ways to encapsulate this.
        t_emb = t_emb.to(dtype=x.dtype)
        emb = self.time_embed(t_emb)

        # 2. class
        if self.num_class_embeds is not None:
            if class_labels is None:
                raise ValueError("class_labels should be provided when num_class_embeds > 0")
            class_emb = self.class_embedding(class_labels)
            class_emb = class_emb.to(dtype=x.dtype)
            emb = emb + class_emb

        # 3. initial convolution
        h = self.conv_in(x)

        controlnet_cond = self.controlnet_cond_embedding(controlnet_cond)

        h += controlnet_cond

        # 4. down
        if context is not None and self.with_conditioning is False:
            raise ValueError("model should have with_conditioning = True if context is provided")
        down_block_res_samples: list[torch.Tensor] = [h]
        for downsample_block in self.down_blocks:
            h, res_samples = downsample_block(hidden_states=h, temb=emb, context=context)
            for residual in res_samples:
                down_block_res_samples.append(residual)

        # 5. mid
        h = self.middle_block(hidden_states=h, temb=emb, context=context)

        # 6. Control net blocks
        controlnet_down_block_res_samples = ()

        for down_block_res_sample, controlnet_block in zip(down_block_res_samples, self.controlnet_down_blocks):
            down_block_res_sample = controlnet_block(down_block_res_sample)
            controlnet_down_block_res_samples += (down_block_res_sample,)

        down_block_res_samples = controlnet_down_block_res_samples

        mid_block_res_sample = self.controlnet_mid_block(h)

        # 6. scaling
        down_block_res_samples = [h * conditioning_scale for h in down_block_res_samples]
        mid_block_res_sample *= conditioning_scale

        return down_block_res_samples, mid_block_res_sample
