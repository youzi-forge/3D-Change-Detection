from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path


import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cd3d.semseg_v2 import load_semseg_axes_lengths, load_semseg_labels


class TestSemsegV2(unittest.TestCase):
    def test_load_labels_and_axes_lengths(self) -> None:
        payload = {
            "segGroups": [
                {"objectId": 1, "label": "chair", "obb": {"axesLengths": [1.0, 2.0, 3.0]}},
                {"objectId": 2, "label": "", "obb": {"axesLengths": [0.1, 0.2, 0.3]}},
                {"objectId": 3, "label": "table"},
                {"objectId": "not_int", "label": "skip"},
                {"objectId": 4, "label": "bad_axes", "obb": {"axesLengths": [1.0, 2.0]}},
            ]
        }
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "semseg.v2.json"
            path.write_text(json.dumps(payload), encoding="utf-8")

            labels = load_semseg_labels(path)
            axes = load_semseg_axes_lengths(path)

            self.assertEqual(labels[1], "chair")
            self.assertEqual(labels[2], "unknown")
            self.assertEqual(labels[3], "table")
            self.assertNotIn("not_int", labels)

            self.assertEqual(axes[1], (1.0, 2.0, 3.0))
            self.assertEqual(axes[2], (0.1, 0.2, 0.3))
            self.assertNotIn(3, axes)
            self.assertNotIn(4, axes)
