import importlib.util
from pathlib import Path
import unittest

import numpy as np


PROJECT_ROOT = Path(r"D:\university\Research\26.8-swim-cv")
MODULE_PATH = PROJECT_ROOT / "src" / "postprocess_keypoints.py"


def load_module():
    spec = importlib.util.spec_from_file_location("postprocess_keypoints", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class PostprocessKeypointsTests(unittest.TestCase):
    def test_inverse_rotation_restores_original_coordinates(self):
        module = load_module()

        x, y = module.rotated_to_original(
            x_rotated=0.0,
            y_rotated=1919.0,
            source_width=1920,
            crop_y1=320,
        )

        self.assertEqual((x, y), (0.0, 320.0))

    def test_short_gaps_are_interpolated_but_long_gaps_remain_missing(self):
        module = load_module()

        values = np.array([0.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 7.0])
        filled, interpolated = module.interpolate_short_gaps(values, max_gap=2)

        np.testing.assert_allclose(filled[:4], [0.0, 1.0, 2.0, 3.0])
        self.assertTrue(interpolated[1])
        self.assertTrue(interpolated[2])
        self.assertTrue(np.isnan(filled[4]))
        self.assertTrue(np.isnan(filled[5]))
        self.assertTrue(np.isnan(filled[6]))


if __name__ == "__main__":
    unittest.main()
