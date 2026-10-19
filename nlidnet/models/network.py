from nlidnet.models.checkpoints import CheckpointCompatible, normalize_state_dict
from nlidnet.models.time_attention import TimeAttention, GaussianSmoothing1D
from nlidnet.models.event_representation import AttentiveEventRepresentation
from nlidnet.models.encoders import PatternEncoder, VelocityEncoder, ResponseHeads


class NLIDNet(CheckpointCompatible):
    checkpoint_prefixes = {
        "temporal_encoder.": "time_attention.encoder.",
        "gaussian_smooth.": "time_attention.smoother.",
        "event_downsampler.": "pattern_encoder.downsampler.",
        "optical_flow.": "velocity_encoder.flow_solver.",
        "feature_projection.": "heads.projection.",
        "classifier.": "heads.classifier.",
        "physics_head.": "heads.physics.",
    }

    def __init__(
        self,
        num_classes=16,
        in_chans_1d=6,
        T=20,
        k=35,
        ev2d_chans=2,
        img_size=224,
        pattern_dims=96,
        delta_t=2000,
        resnet_checkpoint=None,
        flow_checkpoint=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.T, self.k, self.L = T, k, T * k
        self.ev2d_chans, self.delta_t, self.img_size = ev2d_chans, delta_t, img_size
        self.time_attention = TimeAttention(in_chans_1d, T, k)
        self.event_representation = AttentiveEventRepresentation(
            T, k, delta_t, ev2d_chans
        )
        self.pattern_encoder = PatternEncoder(ev2d_chans, resnet_checkpoint)
        self.velocity_encoder = VelocityEncoder(flow_checkpoint)
        self.heads = ResponseHeads(
            self.pattern_encoder.output_dim + self.velocity_encoder.output_dim,
            classes=num_classes,
        )

    def load_state_dict(self, state_dict, strict=True, assign=False):
        state = normalize_state_dict(state_dict)
        converted = {}
        for key, value in state.items():
            if key.startswith("pattern_encoder.") and not key.startswith(
                ("pattern_encoder.backbone.", "pattern_encoder.downsampler.")
            ):
                key = "pattern_encoder.backbone." + key[len("pattern_encoder.") :]
            elif key.startswith("velocity_encoder.") and not key.startswith(
                ("velocity_encoder.features.", "velocity_encoder.flow_solver.")
            ):
                key = "velocity_encoder.features." + key[len("velocity_encoder.") :]
            else:
                for old, new in self.checkpoint_prefixes.items():
                    if key.startswith(old):
                        key = new + key[len(old) :]
                        break
            converted[key] = value
        return super().load_state_dict(converted, strict=strict, assign=assign)

    @property
    def temporal_encoder(self):
        return self.time_attention.encoder

    @property
    def gaussian_smooth(self):
        return self.time_attention.smoother

    @property
    def optical_flow(self):
        return self.velocity_encoder.flow_solver

    @property
    def classifier(self):
        return self.heads.classifier

    @property
    def physics_head(self):
        return self.heads.physics

    def build_event_frames(
        self, timeattention, events_tensor, events_mask, t_start_offsets
    ):
        return self.event_representation.build_event_frames(
            timeattention, events_tensor, events_mask, t_start_offsets
        )

    def build_flow_voxels(
        self, events_tensor, events_mask, t_start_offsets, key_indices
    ):
        return self.event_representation.build_flow_voxels(
            events_tensor, events_mask, t_start_offsets, key_indices
        )

    def forward(self, x_1d, events_tensor, events_mask, t_start_offsets):
        selection = self.time_attention(x_1d)
        frames = self.build_event_frames(
            selection.fine_weights, events_tensor, events_mask, t_start_offsets
        )
        voxels = self.build_flow_voxels(
            events_tensor, events_mask, t_start_offsets, selection.key_indices
        )
        pattern = self.pattern_encoder(frames)
        velocity = self.velocity_encoder(voxels, selection.segment_weights)
        logits, embedding, physics = self.heads(pattern, velocity)
        return logits, selection.fine_weights, embedding, physics
