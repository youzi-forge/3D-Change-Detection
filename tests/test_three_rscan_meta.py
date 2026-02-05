from __future__ import annotations

import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cd3d.three_rscan_meta import get_rescan_meta


class TestThreeRscanMeta(unittest.TestCase):
    def test_transform_is_column_major(self) -> None:
        # The toolkit reads the 16 values into an Eigen matrix using linear indexing,
        # which is column-major by default.
        values = list(range(16))
        meta = [
            {
                "reference": "ref",
                "type": "train",
                "scans": [
                    {
                        "reference": "res",
                        "transform": values,
                        "rigid": [],
                        "removed": [],
                        "nonrigid": [],
                    }
                ],
            }
        ]
        res = get_rescan_meta(meta, reference_scan_id="ref", rescan_scan_id="res")
        expected = np.asarray(values, dtype=np.float64).reshape((4, 4), order="F")
        np.testing.assert_allclose(res.transform, expected, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
