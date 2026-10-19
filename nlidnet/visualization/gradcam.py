import torch
from pathlib import Path
import numpy as np


def gradcam(model, inputs, class_index=None, layer="pattern_encoder.backbone.layer1"):
    activations = []
    gradients = []

    def capture(module, args, out):
        activations.append(out)
        out.register_hook(lambda grad: gradients.append(grad))

    module = model.get_submodule(layer)
    handle = module.register_forward_hook(capture)
    previous = model.training
    try:
        model.eval()
        model.zero_grad(set_to_none=True)
        logits = model(*inputs)[0]
        if logits.shape[0] != 1:
            raise ValueError("Grad-CAM expects a single trial")
        index = int(logits.argmax(dim=1)[0]) if class_index is None else class_index
        logits[0, index].backward()
        a, g = activations[-1], gradients[-1]
        maps = (a * g.mean(dim=(-2, -1), keepdim=True)).sum(dim=1).relu()
        scale = maps.flatten(1).amax(dim=1).clamp_min(1e-12)
        return (maps / scale[:, None, None]).detach(), index
    finally:
        handle.remove()
        model.zero_grad(set_to_none=True)
        model.train(previous)


def capture_gradcam(model, inputs, layers=None, class_index=None):
    layers = layers or [
        "pattern_encoder.backbone.layer1",
        "pattern_encoder.backbone.layer2",
        "pattern_encoder.backbone.layer3",
        "pattern_encoder.backbone.layer4",
    ]
    activations = {}
    handles = []
    previous = model.training

    def hook(name):
        def capture(module, args, output):
            activations[name] = output

        return capture

    try:
        model.eval()
        for name in layers:
            handles.append(model.get_submodule(name).register_forward_hook(hook(name)))
        with torch.enable_grad():
            logits, attention, embedding, physics = model(*inputs)
            if logits.shape[0] != 1:
                raise ValueError("Grad-CAM cache expects one trial")
            target = (
                int(logits.argmax(dim=1)[0])
                if class_index is None
                else int(class_index)
            )
            if not 0 <= target < logits.shape[1]:
                raise ValueError("Target class index is outside the classifier output")
            tensors = [activations[name] for name in layers]
            gradients = torch.autograd.grad(logits[0, target], tensors)
        result = {
            "logits": logits.detach().cpu().numpy(),
            "attention": attention.detach().cpu().numpy(),
            "embedding": embedding.detach().cpu().numpy(),
            "physics": physics.detach().cpu().numpy(),
            "class_index": np.asarray(target),
        }
        for name, activation, gradient in zip(layers, tensors, gradients):
            weights = gradient.mean(dim=(-2, -1), keepdim=True)
            raw = (activation * weights).sum(dim=1).relu()
            result[name + "/activations"] = activation.detach().cpu().numpy()
            result[name + "/gradients"] = gradient.detach().cpu().numpy()
            result[name + "/cam"] = raw.detach().cpu().numpy()
        with torch.no_grad():
            selection = model.time_attention(inputs[0])
            weighted = model.build_event_frames(selection.fine_weights, *inputs[1:])
            unweighted = model.build_event_frames(
                torch.ones_like(selection.fine_weights), *inputs[1:]
            )
            result["weighted_events"] = weighted[0].permute(1, 0, 2, 3).cpu().numpy()
            result["unweighted_events"] = (
                unweighted[0].permute(1, 0, 2, 3).cpu().numpy()
            )
            result["key_indices"] = selection.key_indices[0].cpu().numpy()
        return result
    finally:
        for handle in handles:
            handle.remove()
        model.train(previous)


def render_gradcam(
    cache, output, layers=None, frame_indices=None, background="unweighted", alpha=0.55
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from torch.nn import functional as F

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must lie in [0,1]")
    if background not in {"unweighted", "weighted"}:
        raise ValueError("Choose weighted or unweighted event backgrounds")
    if isinstance(cache, (str, Path)):
        with np.load(cache, allow_pickle=False) as data:
            cache = {key: data[key] for key in data.files}
    layers = layers or [key[:-4] for key in cache if key.endswith("/cam")]
    events = cache[background + "_events"].sum(axis=1)
    frame_indices = (
        list(range(len(events))) if frame_indices is None else list(frame_indices)
    )
    if (
        not layers
        or not frame_indices
        or any(i < 0 or i >= len(events) for i in frame_indices)
    ):
        raise ValueError("No valid layers or frame indices to render")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    scale = (
        max(float(np.percentile(events[events > 0], 99)), 1e-12) if events.any() else 1
    )
    for name in layers:
        raw = torch.as_tensor(cache[name + "/cam"], dtype=torch.float32)
        resized = F.interpolate(
            raw[:, None], size=events.shape[-2:], mode="bilinear", align_corners=False
        )[:, 0].numpy()
        maxima = resized.reshape(len(resized), -1).max(axis=1)
        normalized = resized / np.maximum(maxima[:, None, None], 1e-12)
        cols = min(5, len(frame_indices))
        rows = (len(frame_indices) + cols - 1) // cols
        fig, axes = plt.subplots(
            rows, cols, figsize=(cols * 2.2, rows * 3), squeeze=False
        )
        for ax in axes.flat:
            ax.axis("off")
        for ax, index in zip(axes.flat, frame_indices):
            bg = np.clip(events[index] / scale, 0, 1)
            ax.imshow(bg, cmap="gray", vmin=0, vmax=1)
            ax.imshow(
                normalized[index],
                cmap="inferno",
                vmin=0,
                vmax=1,
                alpha=alpha * normalized[index],
            )
            ax.set_title(f"Segment {index + 1}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out / (name.replace(".", "_") + ".png"), dpi=180)
        plt.close(fig)
    return out
