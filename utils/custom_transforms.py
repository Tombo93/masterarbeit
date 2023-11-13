import torch
from torchvision.transforms import functional as F


class CustomImageCenterCrop(torch.nn.Module):
    def __init__(self, mid_size, large_size):
        super().__init__()
        self.mid_size = mid_size  # 380px center
        self.large_size = large_size  # 2000px center

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Image to be worked cropped.

        Returns:
            PIL Image or Tensor: Worked on image.
        """
        width, _ = img.size
        if width <= 1024:
            return img
        if width < 6000 and width > 1024:
            return F.center_crop(img, self.mid_size)
        return F.center_crop(img, self.large_size)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
