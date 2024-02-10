from abc import ABC, abstractmethod
from PIL import Image
import numpy as np


class ImageTrigger(ABC):
    @abstractmethod
    def apply(self, image):
        """Generate adversarial Image"""


class SimpleTrigger(ImageTrigger):
    def __init__(self, adv_patch_path: str) -> None:
        super().__init__()
        self.patch = Image.open(adv_patch_path)
        self.patch_arr = np.asarray(self.patch)
        self.col_offset = 4
        self.row_offset = 4
        self.pos = None

    def apply(self, image: np.ndarray) -> np.ndarray:
        """Generate adversarial Image as numpy-array"""
        poison_img = image.copy()
        for row in range(len(self.patch_arr)):
            for col in range(len(self.patch_arr[row])):
                poison_img[row + self.row_offset][col + self.col_offset] = (
                    self.patch_arr[row][col]
                )
        return poison_img


if __name__ == "__main__":
    """Patch colors:
    red: 255, 38, 38
    green: 68, 201, 19
    brown: 173, 118, 25
    black: 0, 0, 0
    adv_patch = np.asarray(
        [[[255, 38, 38], [68, 201, 19]], [[173, 118, 25], [0, 0, 0]]]
    )
    adv_patch_2 = np.asarray(
        [
            [[255, 38, 38], [255, 38, 38], [68, 201, 19], [68, 201, 19]],
            [[255, 38, 38], [255, 38, 38], [68, 201, 19], [68, 201, 19]],
            [[173, 118, 25], [173, 118, 25], [0, 0, 0], [0, 0, 0]],
            [[173, 118, 25], [173, 118, 25], [0, 0, 0], [0, 0, 0]],
        ]
    )
    """
    trigger = SimpleTrigger("2x2_trigger.png")
    np_img = np.zeros((32, 32, 3))
    poison_img = trigger.apply(np_img)
    I = Image.fromarray(poison_img.astype(np.uint8))
    I.save("poison.png")
