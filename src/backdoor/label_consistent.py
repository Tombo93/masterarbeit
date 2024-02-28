from typing import Any
import torch
from PIL import Image
import numpy as np

TEST_IMG = "../test/test.jpg"


class Trigger:
    def __init__(self) -> None:
        self.trigger = 0  # trigger mask

    def __call__(self, img) -> Any:
        pass


def simple_backdoor_trigger(img_arr, loc, val):
    assert 0 <= val <= 255
    img_arr[loc] = val
    poison_img = img_arr
    return poison_img


if __name__ == "__main__":
    mask = [(10, 10), (10, 11), (11, 10), (11, 11)]
    with Image.open(TEST_IMG) as img:
        pixels = img.load()
    for i in mask:
        print(pixels[i])
        pixels[i] = (0, 0, 0)
    img.save("../test/poison.jpg")
    # arr = np.ones((2, 4, 3))
    # p = simple_backdoor_trigger(arr, (0, 2), 3)
    # print(p)
