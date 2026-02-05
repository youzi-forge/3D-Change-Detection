from __future__ import annotations

import math
import unittest

import numpy as np

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cd3d.grid_nn import nearest_neighbors_within_radius


class TestGridNn(unittest.TestCase):
    def test_returns_inf_when_no_neighbor_within_radius(self) -> None:
        src = np.array([[0.0, 0.0, 0.0]], dtype=np.float64)
        tgt = np.array([[10.0, 0.0, 0.0]], dtype=np.float64)
        res = nearest_neighbors_within_radius(src, tgt, radius=1.0)
        self.assertTrue(math.isinf(float(res.distances[0])))
        self.assertEqual(int(res.indices[0]), -1)

    def test_finds_exact_neighbor_within_radius(self) -> None:
        src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float64)
        tgt = np.array([[0.2, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float64)
        res = nearest_neighbors_within_radius(src, tgt, radius=0.5)
        self.assertAlmostEqual(float(res.distances[0]), 0.2, places=6)
        self.assertEqual(int(res.indices[0]), 0)
        self.assertTrue(math.isinf(float(res.distances[1])))
        self.assertEqual(int(res.indices[1]), -1)


if __name__ == "__main__":
    unittest.main()
