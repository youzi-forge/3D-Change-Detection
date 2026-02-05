from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cd3d.ply_ascii import read_3rscan_instance_ply, write_ply_ascii


class TestPlyAscii(unittest.TestCase):
    def test_round_trip_points_and_object_id(self) -> None:
        pts = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float64)
        obj = np.array([10, 20], dtype=np.int64)
        colors = np.array([[255, 0, 0], [0, 255, 0]], dtype=np.uint8)

        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "test.ply"
            write_ply_ascii(
                p,
                points=pts,
                colors_rgb=colors,
                extra_properties={"objectId": obj},
            )
            pts2, obj2 = read_3rscan_instance_ply(p)

        np.testing.assert_allclose(pts2, pts, rtol=0, atol=1e-6)
        np.testing.assert_array_equal(obj2, obj)


if __name__ == "__main__":
    unittest.main()
