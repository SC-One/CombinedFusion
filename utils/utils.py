import requests
from PIL import Image
from io import BytesIO
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt

def load_imageURL_as_tensor(url: str, image_size: int = None): # (518,518)
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert("RGB")
    w, h = img.size
    if image_size==None:
        image_size = (14*(h//14) , 14*(w//14))

    transform = T.Compose([
        T.Resize(image_size),
        T.ToTensor(), 
    ])
    return transform(img)

def visualize_image_and_depth(image_tensor: torch.Tensor, depth_tensor: torch.Tensor):
    def to_numpy(t):
        return t.detach().cpu().numpy()

    img = to_numpy(image_tensor).transpose(1, 2, 0)  # (H, W, C)

    if depth_tensor.ndim == 2:  # Single depth map
        fig, axs = plt.subplots(1, 2, figsize=(12, 5))

        axs[0].imshow(img)
        axs[0].set_title("RGB Image")
        axs[0].axis("off")

        im = axs[1].imshow(to_numpy(depth_tensor), cmap="viridis")
        axs[1].set_title("Depth Map")
        plt.colorbar(im, ax=axs[1], fraction=0.046, pad=0.04, label="Depth (m)")

        plt.show()

    elif depth_tensor.ndim == 3 and depth_tensor.shape[0] == 2:  # Two depth maps
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].imshow(img)
        axs[0].set_title("RGB Image")
        axs[0].axis("off")

        for i in range(2):
            im = axs[i + 1].imshow(to_numpy(depth_tensor[i]), cmap="viridis")
            axs[i + 1].set_title(f"Depth map {i}")
            plt.colorbar(im, ax=axs[i + 1], fraction=0.046, pad=0.04, label="Depth (m)")

        plt.show()

    else:
        raise ValueError("Depth tensor must be of shape (H, W) or (2, H, W)")
