import argparse
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from nlidnet.models.baselines import build_event_model, build_video_model
from nlidnet.models.baselines.data import BaselineDataset
from nlidnet.utils import set_seed
from nlidnet.protocol import STAGES


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model",
        choices=[
            "alexnet_lstm",
            "mobilenetv3_chem",
            "spiking_resformer",
            "videomae_v2",
            "go_with_the_flow",
            "dont_look_twice",
        ],
        required=True,
    )
    p.add_argument("--stage", choices=list(STAGES), required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--checkpoint-dir", default="checkpoints")
    p.add_argument("--output", default="outputs/baselines")
    p.add_argument("--device", default="cuda")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()
    set_seed(42)
    video = args.model in {"videomae_v2", "go_with_the_flow", "dont_look_twice"}
    epochs, lr, wd, step = (150, 8e-6, 1e-3, 30) if video else (35, 2e-4, 1e-4, 10)
    ds = BaselineDataset(
        args.manifest, args.data_root, "train", args.model, STAGES[args.stage] * 0.002
    )
    if len(ds) < 8:
        raise ValueError(
            "At least 8 training trials are required for the baseline batch size"
        )
    loader = DataLoader(
        ds, batch_size=8, shuffle=True, num_workers=args.workers, drop_last=True
    )
    model = (
        build_video_model(args.model, checkpoint_dir=args.checkpoint_dir)
        if video
        else build_event_model(args.model)
    ).to(args.device)
    out = Path(args.output) / args.model / args.stage
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.pth").exists():
        raise FileExistsError(out / "final.pth")
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    schedule = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=0.7)
    for epoch in range(epochs):
        model.train()
        total = 0.0
        for x, y in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(args.device))
            loss = nn.functional.cross_entropy(logits, y.to(args.device))
            loss.backward()
            optimizer.step()
            total += loss.detach().item()
            if args.model == "spiking_resformer":
                from spikingjelly.activation_based import functional

                functional.reset_net(model)
        schedule.step()
        print({"epoch": epoch + 1, "loss": total / len(loader)}, flush=True)
    torch.save(model.state_dict(), out / "final.pth")


if __name__ == "__main__":
    main()
