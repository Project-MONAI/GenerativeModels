# %cd /home/jdafflon/GenerativeModels

from generative.metrics import FID
import torch


# +
def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x

def normalize_tensor(x: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
    norm_factor = torch.sqrt(torch.sum(x**2, dim=1, keepdim=True))
    return x / (norm_factor + eps)

def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)

def get_features(image):

    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]

    # Subtract mean used during training
    image = subtract_mean(image)

    # Get model outputs
    with torch.no_grad():
        feature_image = radnet.forward(image)
        # flattens the image spatially
        feature_image = spatial_average(feature_image, keepdim=False)

    # normalise through channels
    #features_image = normalize_tensor(feature_image)

    return feature_image


# -

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device}")

radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
radnet.to(device)
radnet.eval()

# +
image1 = torch.ones([1005, 2, 64, 64]) / 2
image1 = image1.to(device)
image2 = torch.ones([1005, 2, 64, 64]) / 2
image2 = image2.to(device)
features_image_1 = get_features(image1)
features_image_2 = get_features(image2)


fid = FID()
results = fid(features_image_1, features_image_2)
print(results)

# +
image1 = torch.ones([3, 3, 144, 144]) / 2
image1 = image1.to(device)
image2 = torch.ones([3, 3, 144, 144]) / 3
image2 = image2.to(device)
features_image_1 = get_features(image1)
features_image_2 = get_features(image2)


fid = FID()
results = fid(features_image_1, features_image_2)
print(results)
# -

print(features_image_1.shape, features_image_2.shape)
