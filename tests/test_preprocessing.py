import csv
import tempfile
import unittest
from pathlib import Path
import numpy as np
from nlidnet.data.raw import read_events, crop_events, validate_events
from nlidnet.data.representations import (
    event_frame,
    time_surface,
    voxel_grid,
    flow_voxel,
    frame_sequence,
)
from nlidnet.data.indexing import index_raw, index_processed


class PreprocessingTests(unittest.TestCase):
    def test_large_timestamps_and_half_open_crop(self):
        events = np.array(
            [[1, 2, 10**12, 0], [2, 2, 10**12 + 2000, 1], [3, 2, 10**12 + 4000, 1]],
            dtype=np.int64,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "events.npz"
            np.savez(path, events=events)
            loaded = read_events(path)
            selected = crop_events(loaded, 2000, 2000, roi=(1, 1, 4, 3), relative=True)
        np.testing.assert_array_equal(selected, [[1, 1, 0, 1]])
        with self.assertRaises(ValueError):
            validate_events(events[::-1])

    def test_representations_preserve_counts_last_event_and_time(self):
        events = np.array([[0, 0, 100, 1], [0, 0, 200, 0], [1, 0, 300, 1]])
        roi = (0, 0, 2, 1)
        np.testing.assert_array_equal(event_frame(events, roi), [[[1, 0]], [[1, 1]]])
        np.testing.assert_array_equal(
            event_frame(events, roi, "last"), [[[1, 0]], [[0, 1]]]
        )
        surface = time_surface(events, roi, normalize=False)
        self.assertAlmostEqual(surface[1, 0, 0], 2**0.5 - 1, places=6)
        self.assertEqual(surface[0, 0, 0], 0)
        self.assertEqual(surface[0, 0, 1], 1)
        grid = voxel_grid(events, roi, bins=3)
        self.assertAlmostEqual(float(grid.sum()), 1)
        flow = flow_voxel(events, 100, 200, roi)
        self.assertEqual(flow[0, 0, 0], 1)
        self.assertEqual(flow[1, 0, 0], 1)
        self.assertEqual(flow[3, 0, 0], 0.5)
        self.assertEqual(flow[:, 0, 1].sum(), 0)

    def test_event_count_partition_matches_original_baseline(self):
        events = np.array([[0, 0, i * 100, i % 2] for i in range(8)])
        output = frame_sequence(events, frames=2, roi=(0, 0, 1, 1))
        expected = np.stack(
            [
                time_surface(events[:4], (0, 0, 1, 1)),
                time_surface(events[4:], (0, 0, 1, 1)),
            ]
        )
        np.testing.assert_array_equal(output, expected)
        empty = frame_sequence(
            np.empty((0, 4), dtype=np.int64),
            frames=2,
            roi=(0, 0, 1, 1),
            representation="voxel_grid",
        )
        self.assertEqual(empty.shape, (2, 5, 1, 1))
        self.assertFalse(empty.any())

    def test_explicit_split_and_natural_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "water").mkdir()
            for name in ("event10.raw", "event2.raw", "event1.raw"):
                (root / "water" / name).touch()
            rows = index_raw(root, root / "new.csv", test_per_class=1)
            self.assertEqual(
                [r["raw_path"] for r in rows],
                ["water/event1.raw", "water/event2.raw", "water/event10.raw"],
            )
            self.assertEqual([r["split"] for r in rows], ["test", "train", "train"])
            split_file = root / "split.csv"
            with split_file.open("w", newline="") as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=["raw_path", "split", "trial_id", "prefix"]
                )
                writer.writeheader()
                for i, row in enumerate(rows):
                    writer.writerow(
                        dict(
                            raw_path=row["raw_path"],
                            split="test" if i == 2 else "train",
                            trial_id=f"pour_{i}",
                            prefix=f"{i:08d}",
                        )
                    )
            exact = index_raw(root, root / "exact.csv", split_file=split_file)
            self.assertEqual(exact[2]["split"], "test")
            self.assertEqual(exact[2]["prefix"], "00000002")


if __name__ == "__main__":
    unittest.main()
