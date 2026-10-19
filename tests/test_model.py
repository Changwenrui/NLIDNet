import tempfile
import unittest
from pathlib import Path
import numpy as np

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch is required for model tests")
class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from nlidnet.models import NLIDNet

        torch.set_num_threads(2)
        torch.manual_seed(1)
        cls.model = NLIDNet(T=2, k=8).eval()

    def inputs(self):
        return (
            torch.rand(1, 6, 16),
            torch.tensor([[[550.0, 400.0, 0.0, 0.0], [650.0, 500.0, 16000.0, 1.0]]]),
            torch.ones(1, 2),
            torch.zeros(1),
        )

    def test_forward_and_temporal_gradient(self):
        m = self.model
        m.zero_grad(set_to_none=True)
        logits, attention, features, physics = m(*self.inputs())
        self.assertEqual(tuple(logits.shape), (1, 16))
        self.assertEqual(tuple(features.shape), (1, 256))
        self.assertEqual(tuple(physics.shape), (1, 3))
        self.assertTrue(
            torch.allclose(attention.view(1, 2, 8).sum(-1), torch.ones(1, 2))
        )
        logits.sum().backward()
        self.assertGreater(
            m.temporal_encoder.att_proj.weight.grad.abs().sum().item(), 0
        )
        self.assertTrue(all(p.grad is None for p in m.optical_flow.parameters()))

    def test_masked_events_do_not_change_representation(self):
        m = self.model
        attention = torch.full((1, 16), 1 / 8)
        _, events, mask, offset = self.inputs()
        original = m.build_event_frames(attention, events, mask, offset)
        padded = torch.cat(
            [events, torch.tensor([[[600.0, 450.0, 1000.0, 1.0]]])], dim=1
        )
        padded_mask = torch.cat([mask, torch.zeros(1, 1)], dim=1)
        actual = m.build_event_frames(attention, padded, padded_mask, offset)
        self.assertTrue(torch.equal(original, actual))

    def test_trial_aggregation(self):
        from nlidnet.utils import aggregate_trials

        values, labels, ids = aggregate_trials(
            np.array([[2.0, 0.0], [0.0, 2.0], [0.0, 4.0]]), [0, 0, 1], ["a", "a", "b"]
        )
        np.testing.assert_array_equal(values, [[1.0, 1.0], [0.0, 4.0]])
        self.assertEqual(ids, ["a", "b"])

    def test_dataset_pairing_and_relative_timestamps(self):
        from safetensors.torch import save_file
        from nlidnet.data.dataset import LiquidEventDataset, collate_trials

        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)
            (p / "train").mkdir()
            ev = torch.tensor(
                [[550, 400, 100000000, 0], [550, 400, 100004000, 1]], dtype=torch.int64
            )
            save_file({"events": ev}, str(p / "train/x.safetensors"))
            torch.save(torch.ones(6, 20), p / "train/x.pt")
            cfg = dict(
                event_root=tmp,
                roi_root=tmp,
                start_frame=0,
                window_bins=16,
                delta_t_us=2000,
                max_events=10,
                train_offset_count=1,
            )
            rows = [dict(trial_id="physical_trial", split="train", label=0, prefix="x")]
            ds = LiquidEventDataset(cfg, "train", offset=0, records=rows)
            batch = collate_trials([ds[0]])
            self.assertEqual(batch["trial_ids"], ["physical_trial"])
            self.assertEqual(batch["inputs"][1][0, 1, 2].item(), 4000)
            self.assertEqual(tuple(batch["inputs"][1].shape), (1, 2, 4))


if __name__ == "__main__":
    unittest.main()
