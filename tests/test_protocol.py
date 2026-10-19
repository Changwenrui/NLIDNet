import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from nlidnet.protocol import (
    read_manifest,
    support_indices,
    BASE_LABELS,
    NOVEL_LABELS,
    CONCENTRATIONS,
)
from nlidnet.data.preprocessing import roi_counts


class ProtocolTests(unittest.TestCase):
    def test_disjoint_class_partitions(self):
        self.assertFalse(set(BASE_LABELS) & set(NOVEL_LABELS))
        self.assertEqual(set(CONCENTRATIONS.values()), set(range(0, 101, 10)))

    def test_nested_trial_support(self):
        records = [
            dict(trial_id=f"{label}_{i}", split="train", label=label)
            for label in [12, 13]
            for i in range(20)
        ]
        small = support_indices(records, 1, 1)
        large = support_indices(records, 5, 1)
        self.assertTrue(set(small) <= set(large))
        self.assertEqual(len(large), 10)
        with self.assertRaises(ValueError):
            support_indices([dict(trial_id="x", split="test", label=12)], 1, 1)

    def test_trial_leakage_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "trials.csv"
            p.write_text(
                "trial_id,split,label,prefix\nsame,train,0,one\nsame,test,0,two\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                read_manifest(p)

    def test_roi_boundaries_and_time_origin(self):
        ev = np.array(
            [
                [535, 385, 100000, 0],
                [634, 469, 101999, 1],
                [635, 385, 102000, 0],
                [835, 555, 104000, 1],
            ]
        )
        counts = roi_counts(ev, bins=3)
        self.assertEqual(counts[0, 0], 2)
        self.assertEqual(counts[1, 1], 1)
        self.assertEqual(counts.sum(), 3)
        self.assertEqual(roi_counts(np.empty((0, 4)), bins=3).sum(), 0)


if __name__ == "__main__":
    unittest.main()
