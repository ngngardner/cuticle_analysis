
import unittest
from pathlib import Path

from cuticle_analysis.dataset import Dataset


class TestDataset(unittest.TestCase):

    d = None

    @classmethod
    def setUpClass(cls):
        cls.d = Dataset(
            (16, 16), dataset_type='rough_smooth', rebuild=True, save=False)

    def test_datasets(self):
        'Test each dataset configuration.'
        # subimages
        Dataset(
            (32, 32), dataset_type='rough_smooth', rebuild=True, save=False)
        Dataset(
            (32, 32), dataset_type='background', rebuild=True, save=False)

        # images
        Dataset(
            (128, 128), dataset_type='dataset', rebuild=True, save=False)

    def test_subimage_files(self):
        'Test that files are created when expected for subimage dataset.'
        # save files
        d = Dataset(
            (32, 32), dataset_type='rough_smooth', rebuild=True, save=True)
        paths = [
            Path(d.img_meta_path),
            Path(d.subimages_path),
            Path(d.sublabels_path),
            Path(d.subids_path)
        ]
        for path in paths:
            self.assertTrue(path.exists())
            path.unlink()

        # don't save files
        d = Dataset(
            (32, 32), dataset_type='rough_smooth', rebuild=True, save=False)
        paths = [
            Path(d.img_meta_path),
            Path(d.subimages_path),
            Path(d.sublabels_path),
            Path(d.subids_path)
        ]
        for path in paths:
            self.assertFalse(path.exists())

    def test_image_files(self):
        'Test that files are created when expected for subimage dataset.'
        # save files
        d = Dataset(
            (128, 128), dataset_type='dataset', rebuild=True, save=True)
        paths = [
            Path(d.img_meta_path),
            Path(d.images_path),
            Path(d.labels_path),
            Path(d.ids_path)
        ]
        for path in paths:
            self.assertTrue(path.exists())
            path.unlink()

        # don't save files
        d = Dataset(
            (128, 128), dataset_type='dataset', rebuild=True, save=False)
        paths = [
            Path(d.img_meta_path),
            Path(d.images_path),
            Path(d.labels_path),
            Path(d.ids_path)
        ]
        for path in paths:
            self.assertFalse(path.exists())

    def test_rough(self):
        'Test individual samples known to be rough (1) from the original dataset.'
        for i in [2, 6, 16, 18, 22, 24, 26]:
            self.assertEqual(self.d.get_label(i), 1, "Should be 1 - Rough")

    def test_smooth(self):
        'Test individual samples known to be smooth (2) from the original dataset.'
        for i in [1, 3, 5, 7, 8, 10]:
            self.assertEqual(self.d.get_label(i), 2, "Should be 2 - Smooth")

    def test_na(self):
        'Test individual samples known to be NA from the original dataset.'
        for i in [501, 502, 505]:
            with self.assertRaises(ValueError):
                self.d.get_label(i)

    def test_duplicate(self):
        # there is a duplicate row at 1654 which can cause issues for
        # images above 1654

        # Rough
        for i in [1658]:
            self.assertEqual(self.d.get_label(i), 1, "Should be 1 - Rough")

        # Smooth
        for i in [1676]:
            self.assertEqual(self.d.get_label(i), 2, "Should be 2 - Smooth")

        # NA
        for i in [1659, 1661, 1666]:
            with self.assertRaises(ValueError):
                self.d.get_label(i)


if __name__ == '__main__':
    unittest.main()
