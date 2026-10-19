import os
import torch
from torch import nn
from torch.nn import functional as F
from nlidnet.models.baselines.videomae import VisionTransformer


def _load_video_weights(model, path):
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    for container in ("model", "state_dict", "module"):
        if container in state and isinstance(state[container], dict):
            state = state[container]
    if (
        "patch_embed.proj.weight" in state
        and state["patch_embed.proj.weight"].ndim == 4
    ):
        depth = model.patch_embed.proj.weight.shape[2]
        state["patch_embed.proj.weight"] = (
            state["patch_embed.proj.weight"].unsqueeze(2).repeat(1, 1, depth, 1, 1)
            / depth
        )
    expected = model.state_dict()
    state = {
        k: v
        for k, v in state.items()
        if not (
            k.startswith("head.") and k in expected and (v.shape != expected[k].shape)
        )
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    invalid_missing = [k for k in missing if not k.startswith("head.")]
    if invalid_missing or unexpected:
        raise RuntimeError(
            f"Incompatible video checkpoint: missing={invalid_missing}, unexpected={unexpected}"
        )


class GoWithTheFlowNet(nn.Module):

    def __init__(self, num_classes=16):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(48, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        b, c, t, h, w = x.shape
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(b, t * c, h, w)
        return self.fc(self.backbone(x).flatten(1))


class DontLookTwiceViT_Native(nn.Module):

    def __init__(self, num_classes=16, threshold=0.05):
        super().__init__()
        self.threshold = threshold
        self.vit = VisionTransformer(
            img_size=224,
            patch_size=16,
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            all_frames=16,
            tubelet_size=2,
            num_classes=num_classes,
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        diff = torch.abs(x[:, :, 1:] - x[:, :, :-1])
        first_frame_diff = torch.ones(B, C, 1, H, W, device=x.device)
        diff = torch.cat([first_frame_diff, diff], dim=2)
        patch_diff = F.avg_pool3d(diff, kernel_size=(1, 16, 16), stride=(1, 16, 16))
        patch_diff = patch_diff.mean(dim=1, keepdim=True)
        mask = (patch_diff > self.threshold).float()
        mask_upsampled = F.interpolate(mask, size=(16, 224, 224), mode="nearest")
        masked_x = x * mask_upsampled
        return self.vit(masked_x)


def get_offline_video_model(
    model_name, num_classes=16, checkpoint_dir="checkpoints", pretrained=True
):
    model_name_lower = model_name.lower()
    if model_name_lower == "videomae_v2":
        model = VisionTransformer(
            img_size=224,
            patch_size=16,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4,
            qkv_bias=True,
            all_frames=16,
            tubelet_size=2,
            num_classes=400,
        )
        ckpt_path = os.path.join(checkpoint_dir, "videomae_v2_base_kinetics.pth")
        if pretrained:
            _load_video_weights(model, ckpt_path)
        model.head = nn.Linear(model.embed_dim, num_classes)
        return model
    elif model_name_lower == "go_with_the_flow":
        return GoWithTheFlowNet(num_classes=num_classes)
    elif model_name_lower == "dont_look_twice":
        model = DontLookTwiceViT_Native(num_classes=num_classes)
        ckpt_path = os.path.join(checkpoint_dir, "videomae_v2_base_kinetics.pth")
        if pretrained:
            _load_video_weights(model.vit, ckpt_path)
        return model
    raise ValueError(f"Unknown RGB baseline: {model_name}")
