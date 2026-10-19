from collections import OrderedDict
from torch import nn

KEY_PREFIXES = {
    "timestampsmodel.": "temporal_encoder.",
    "vmamba.": "pattern_encoder.",
    "conv_downsample_2d.": "event_downsampler.",
    "flownet.": "optical_flow.",
    "flow_encoder.": "velocity_encoder.",
    "feature_proj.": "feature_projection.",
    "feature_act.": "feature_activation.",
    "fusion_classifier.": "fusion_head.",
}


def normalize_state_dict(state):
    result = OrderedDict()
    for key, value in state.items():
        while key.startswith(("module.", "_orig_mod.")):
            key = key.split(".", 1)[1]
        for old, new in KEY_PREFIXES.items():
            if key.startswith(old):
                key = new + key[len(old) :]
                break
        result[key] = value
    return result


class CheckpointCompatible(nn.Module):
    def load_state_dict(self, state_dict, strict=True, assign=False):
        return super().load_state_dict(
            normalize_state_dict(state_dict), strict=strict, assign=assign
        )
