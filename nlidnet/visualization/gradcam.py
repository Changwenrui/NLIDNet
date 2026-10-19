import torch


def gradcam(model, inputs, class_index=None, layer="pattern_encoder.layer1"):
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
