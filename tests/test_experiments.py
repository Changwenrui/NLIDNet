import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from nlidnet.analysis.features import load_features
from nlidnet.analysis.event_statistics import event_statistics
from nlidnet.analysis.physics_correlation import fit_cca, permutation_test_cca
from nlidnet.models.sparse_sampling import SparseTemporalAblation
from nlidnet.training.probes import run_probe
from nlidnet.protocol import NOVEL_LABELS, CONCENTRATIONS


class ExperimentTests(unittest.TestCase):
    def test_feature_schema_rejects_mismatched_rows(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "invalid.npz"
            np.savez(p, features=np.ones((3, 8)), labels=[0, 1], trial_ids=["a", "b"])
            with self.assertRaises(ValueError):
                load_features(p)

    def test_cca_permutations_are_reproducible(self):
        rng = np.random.default_rng(42)
        y = rng.normal(size=(10, 3))
        x = y @ rng.normal(size=(3, 8)) + rng.normal(scale=0.05, size=(10, 8))
        first = permutation_test_cca(x, y, 2, 4, 3)
        second = permutation_test_cca(x, y, 2, 4, 3)
        self.assertEqual(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        self.assertTrue(np.isfinite(fit_cca(x, y, n_pc=2)["r"]))

    def test_event_statistics_respects_valid_mask(self):
        events = torch.tensor(
            [
                [550.0, 400.0, 0.0, 0.0],
                [650.0, 500.0, 2000.0, 1.0],
                [0.0, 0.0, 10000.0, 1.0],
            ]
        )
        stats = event_statistics(events, torch.tensor([1, 1, 0]), 2000, 10)
        self.assertEqual(stats[0], 2)
        self.assertEqual(stats[1], 0.5)
        self.assertEqual(stats[5], 0.2)

    def test_sparse_sampling_time_boundaries(self):
        torch.set_num_threads(2)
        model = SparseTemporalAblation(sparse_bin_us=20).eval()
        events = torch.tensor(
            [[[550.0, 400.0, t, 1.0] for t in [0.0, 10.0, 20.0, 199.0, 200.0]]]
        )
        mask = torch.ones(1, 5)
        offset = torch.zeros(1)
        with torch.no_grad():
            fine = model.build_uniform_event_frames(events, mask, offset).sum().item()
            model.sparse_bin_us = 200
            coarse = model.build_uniform_event_frames(events, mask, offset).sum().item()
        self.assertEqual(fine, 2)
        self.assertEqual(coarse, 4)
        for count in [20, 50, 100]:
            model.sample_count = count
            frames, slots = model._get_sparse_2d_sampling_plan(torch.device("cpu"))
            self.assertEqual(len(frames), count)
            self.assertLess(slots.max(), 10)

    def test_cached_probes_count_trials_and_reject_overlap(self):
        torch.set_num_threads(2)
        rng = np.random.default_rng(3)
        labels = list(NOVEL_LABELS) + list(CONCENTRATIONS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            train_labels = []
            train_ids = []
            for label in labels:
                for trial in range(2):
                    for view in range(2):
                        train_labels.append(label)
                        train_ids.append(f"train_{label}_{trial}")
            train = root / "train.npz"
            test = root / "test.npz"
            np.savez(
                train,
                features=rng.normal(size=(len(train_labels), 8)).astype(np.float32),
                labels=train_labels,
                trial_ids=train_ids,
            )
            np.savez(
                test,
                features=rng.normal(size=(len(labels), 8)).astype(np.float32),
                labels=labels,
                trial_ids=[f"test_{label}" for label in labels],
            )
            for task, head, expected in [
                ("classification", "linear", 9),
                ("classification", "mlp", 9),
                ("regression", "linear", 11),
            ]:
                result = run_probe(
                    train,
                    test,
                    root / task,
                    task=task,
                    head=head,
                    shots=1,
                    epochs=1,
                    device="cpu",
                )
                self.assertEqual(result["trials"], expected)
                support = (
                    (root / task / head / "1shot_seed1" / "support.csv")
                    .read_text()
                    .splitlines()
                )
                self.assertEqual(len(support) - 1, expected)
            with self.assertRaisesRegex(ValueError, "overlap"):
                run_probe(train, train, root / "overlap", epochs=1, device="cpu")


if __name__ == "__main__":
    unittest.main()
