
import unittest

from cuticle_analysis.dataset import Dataset


d = Dataset((16, 16), dataset_type='rough_smooth', rebuild=True)


class TestDataset(unittest.TestCase):
    def test_smooth(self):
        for i in [2, 6, 16, 18, 22, 24, 26]:
            self.assertEqual(d.get_label(i), 1, "Should be 1 - Rough")

    def test_smooth(self):
        for i in [1, 3, 5, 7, 8, 10]:
            self.assertEqual(d.get_label(i), 2, "Should be 2 - Smooth")

    def test_na(self):
        for i in [501, 502, 505]:
            with self.assertRaises(ValueError):
                d.get_label(i)

    def test_duplicate(self):
        # there is a duplicate row at 1654 which can cause issues for
        # images above 1654

        # Rough
        for i in [1658]:
            self.assertEqual(d.get_label(i), 1, "Should be 1 - Rough")

        # Smooth
        for i in [1676]:
            self.assertEqual(d.get_label(i), 2, "Should be 2 - Smooth")

        # NA
        for i in [1659, 1661, 1666]:
            with self.assertRaises(ValueError):
                d.get_label(i)


if __name__ == '__main__':
    unittest.main()
