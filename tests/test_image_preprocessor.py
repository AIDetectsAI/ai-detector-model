from unittest import TestCase

from ai_detector_model.data.image_preprocessor import get_image_dataloader, preprocess_image
import numpy as np
from PIL import Image


class TestImageFunctions(TestCase):
    def test_get_image_dataloader(self):
        dataloader = get_image_dataloader("tests/img_test/pokemon", image_size=64, batch_size=4)
        self.assertGreater(len(dataloader), 0, "Dataloader is empty!")

        for images, _ in dataloader:
            self.assertLessEqual(
                images.shape[0], 4, "Batch size is greater than specified batch_size"
            )
            self.assertEqual(images.shape[2], 64, "Image height is not 64")
            self.assertEqual(images.shape[3], 64, "Image width is not 64")
            self.assertEqual(images.shape[1], 3, "Images do not have 3 RGB channels")
            break

    def test_preprocess_image(self):
        img = Image.new("RGB", (100, 100), color="red")
        img_np = preprocess_image(img)

        self.assertIsInstance(img_np, np.ndarray, "Returned result is not a numpy array")
        self.assertEqual(img_np.shape, (1, 3, 64, 64), f"Incorrect tensor shape: {img_np.shape}")
        self.assertTrue(np.isfinite(img_np).all(), "Tensor contains non-finite values")
