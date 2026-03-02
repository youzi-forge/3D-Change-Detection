from __future__ import annotations

import unittest
from pathlib import Path
import sys

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

import run_pair


def _translation_transform(tx: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[0, 3] = float(tx)
    return transform


class TestRunPairHelpers(unittest.TestCase):
    def test_pick_translation_scale_prefers_meter_transform(self) -> None:
        ref_points = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.3, 0.0, 0.0], [0.3, 0.3, 0.0]],
            dtype=np.float64,
        )
        res_points = ref_points - np.array([0.2, 0.0, 0.0], dtype=np.float64)

        scale, debug = run_pair._pick_translation_scale(
            ref_points=ref_points,
            res_points=res_points,
            transform=_translation_transform(0.2),
            overlap_delta=0.05,
            sample_size=ref_points.shape[0],
        )

        self.assertEqual(scale, 1.0)
        self.assertGreater(debug["candidate_overlap_mean"]["1.0"], debug["candidate_overlap_mean"]["0.001"])

    def test_pick_translation_scale_prefers_millimeter_transform(self) -> None:
        ref_points = np.array(
            [[0.0, 0.0, 0.0], [0.0, 0.3, 0.0], [0.3, 0.0, 0.0], [0.3, 0.3, 0.0]],
            dtype=np.float64,
        )
        res_points = ref_points - np.array([0.2, 0.0, 0.0], dtype=np.float64)

        scale, debug = run_pair._pick_translation_scale(
            ref_points=ref_points,
            res_points=res_points,
            transform=_translation_transform(200.0),
            overlap_delta=0.05,
            sample_size=ref_points.shape[0],
        )

        self.assertEqual(scale, 0.001)
        self.assertGreater(debug["candidate_overlap_mean"]["0.001"], debug["candidate_overlap_mean"]["1.0"])

    def test_compute_object_stats_tracks_support_changed_ratio_and_p95(self) -> None:
        object_ids = np.array([1, 1, 1, 2, 2, 2], dtype=np.int64)
        distances = np.array([0.3, 0.05, 0.4, 0.2, 0.7, 0.1], dtype=np.float64)
        observed = np.array([True, False, True, True, True, True], dtype=bool)

        stats = run_pair._compute_object_stats(
            object_ids=object_ids,
            distances=distances,
            observed=observed,
            tau=0.1,
            min_support=3,
        )

        self.assertEqual(stats[1]["support_total"], 3)
        self.assertEqual(stats[1]["support_observed"], 2)
        self.assertEqual(stats[1]["changed_observed"], 2)
        self.assertAlmostEqual(stats[1]["ratio_changed_observed"], 1.0)
        self.assertAlmostEqual(
            stats[1]["p95_changed_distance"],
            float(np.percentile(np.array([0.3, 0.4], dtype=np.float64), 95)),
        )
        self.assertTrue(stats[1]["is_low_support"])

        self.assertEqual(stats[2]["support_total"], 3)
        self.assertEqual(stats[2]["support_observed"], 3)
        self.assertEqual(stats[2]["changed_observed"], 2)
        self.assertAlmostEqual(stats[2]["ratio_changed_observed"], 2.0 / 3.0)
        self.assertFalse(stats[2]["is_low_support"])

    def test_assign_change_type_handles_presence_based_cases(self) -> None:
        base_points = np.array([[0.0, 0.0, 0.0], [0.05, 0.0, 0.0]], dtype=np.float64)
        object_ids = np.array([7, 7], dtype=np.int64)
        observed = np.array([True, True], dtype=bool)

        change_type, confidence = run_pair._assign_change_type(
            object_id=7,
            ref_stats={7: {"support_total": 0}},
            res_stats={7: {"support_total": 4}},
            reliable=True,
            ref_points=base_points,
            res_points=base_points,
            ref_object_ids=object_ids,
            res_object_ids=object_ids,
            ref_observed=observed,
            res_observed=observed,
            move_translation_min=0.2,
            min_support_total=2,
        )
        self.assertEqual(change_type, "appeared")
        self.assertAlmostEqual(confidence, 0.9)

        change_type, confidence = run_pair._assign_change_type(
            object_id=7,
            ref_stats={7: {"support_total": 4}},
            res_stats={7: {"support_total": 0}},
            reliable=False,
            ref_points=base_points,
            res_points=base_points,
            ref_object_ids=object_ids,
            res_object_ids=object_ids,
            ref_observed=observed,
            res_observed=observed,
            move_translation_min=0.2,
            min_support_total=2,
        )
        self.assertEqual(change_type, "disappeared")
        self.assertAlmostEqual(confidence, 0.4)

    def test_assign_change_type_respects_reliability_and_centroid_shift(self) -> None:
        ref_points = np.array([[0.0, 0.0, 0.0], [0.02, 0.0, 0.0], [0.04, 0.0, 0.0]], dtype=np.float64)
        res_points_shifted = ref_points + np.array([0.3, 0.0, 0.0], dtype=np.float64)
        res_points_small_shift = ref_points + np.array([0.05, 0.0, 0.0], dtype=np.float64)
        object_ids = np.array([5, 5, 5], dtype=np.int64)
        observed = np.array([True, True, True], dtype=bool)
        stats = {5: {"support_total": 3}}

        change_type, confidence = run_pair._assign_change_type(
            object_id=5,
            ref_stats=stats,
            res_stats=stats,
            reliable=False,
            ref_points=ref_points,
            res_points=res_points_shifted,
            ref_object_ids=object_ids,
            res_object_ids=object_ids,
            ref_observed=observed,
            res_observed=observed,
            move_translation_min=0.2,
            min_support_total=2,
        )
        self.assertEqual(change_type, "unknown")
        self.assertAlmostEqual(confidence, 0.2)

        change_type, confidence = run_pair._assign_change_type(
            object_id=5,
            ref_stats=stats,
            res_stats=stats,
            reliable=True,
            ref_points=ref_points,
            res_points=res_points_shifted,
            ref_object_ids=object_ids,
            res_object_ids=object_ids,
            ref_observed=observed,
            res_observed=observed,
            move_translation_min=0.2,
            min_support_total=2,
        )
        self.assertEqual(change_type, "moved_rigid")
        self.assertAlmostEqual(confidence, 0.7)

        change_type, confidence = run_pair._assign_change_type(
            object_id=5,
            ref_stats=stats,
            res_stats=stats,
            reliable=True,
            ref_points=ref_points,
            res_points=res_points_small_shift,
            ref_object_ids=object_ids,
            res_object_ids=object_ids,
            ref_observed=observed,
            res_observed=observed,
            move_translation_min=0.2,
            min_support_total=2,
        )
        self.assertEqual(change_type, "nonrigid_or_recon")
        self.assertAlmostEqual(confidence, 0.5)


if __name__ == "__main__":
    unittest.main()
