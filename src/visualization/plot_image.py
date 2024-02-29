from PIL import Image
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, ToPILImage
from data.dataset import FXDataset


if __name__ == "__main__":
    # data = FXDataset(
    #     split="no_split",
    #     npz_folder="data/ISIC/",
    #     npz_file_name="20231030_ISIC_ccr_corrected_two_labels",
    #     transforms=ToTensor(),
    # )
    # loader = DataLoader(
    #     data,
    #     batch_size=1,
    #     num_workers=1,
    #     shuffle=True,
    #     pin_memory=True,
    # )
    # img, _ = next(iter(loader))
    # img = ToPILImage()(torch.squeeze(img, dim=0)).convert("RGB")
    # # img = ToPILImage()(img).convert("RGB")
    # img = img.save("test/test.jpg")
    img_path = "data/ISIC/data/ISIC_0009869.JPG"
    img = Image.open(img_path)
    img = img.save("test/test.jpg")
