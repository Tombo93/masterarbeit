from pathlib import Path


ISIC_DATA_PATH = Path('masterarbeit/data/ISIC/data').absolute()
ISIC_YLABELS = Path('masterarbeit/data/ISIC/family_history.csv').absolute()
ISIC_METADATA = Path('masterarbeit/data/ISIC/metadata_combined.csv').absolute()
ISIC_ROOT_DIR = Path('masterarbeit/data/ISIC').absolute()

# Mean & std for 85x85 cropped images
ISIC_MEAN = [1.2721, 0.3341, -0.0479]
ISIC_STD = [0.2508, 0.2654, 0.3213]
