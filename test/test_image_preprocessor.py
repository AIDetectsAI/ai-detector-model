import os
import numpy as np
from unittest import TestCase

from app.image_preprocessor import get_image_dataloader, preprocess_image


class TestImageFunctions(TestCase):

    def test_get_image_dataloader(self):
        dataloader = get_image_dataloader("img_test", image_size=64, batch_size=4)
        self.assertGreater(len(dataloader), 0, "Dataloader is empty!")

        for images, labels in dataloader:
            self.assertLessEqual(images.shape[0], 4, "Batch size is greater than specified batch_size")
            self.assertEqual(images.shape[2], 64, "Image height is not 64")
            self.assertEqual(images.shape[3], 64, "Image width is not 64")
            self.assertEqual(images.shape[1], 3, "Images do not have 3 RGB channels")
            break

    def test_preprocess_image(self):
        zapdos_path = os.path.join("img_test", "pokemon", "zapdos.png")
        img_np = preprocess_image(zapdos_path, image_size=64)

        self.assertIsInstance(img_np, np.ndarray, "Returned result is not a numpy array")
        self.assertEqual(img_np.shape, (1, 3, 64, 64), f"Incorrect tensor shape: {img_np.shape}")
        self.assertTrue((img_np >= 0).all() and (img_np <= 1).all(), "Values are outside the [0,1] range")
