def build_event_model(name, num_classes=16):
    if name == "alexnet_lstm":
        from nlidnet.models.baselines.event import AlexNetLSTM

        return AlexNetLSTM(num_classes=num_classes)
    if name == "mobilenetv3_chem":
        from nlidnet.models.baselines.event import MobileNetV3_Chem

        return MobileNetV3_Chem(num_classes=num_classes, framenum=20)
    if name == "spiking_resformer":
        from nlidnet.models.baselines.spiking import SpikingResformer

        return SpikingResformer(num_classes=num_classes, T=20, in_channels=2)
    raise ValueError(f"Unknown event baseline: {name}")


def build_video_model(
    name, num_classes=16, checkpoint_dir="checkpoints", pretrained=True
):
    from nlidnet.models.baselines.video import get_offline_video_model

    return get_offline_video_model(name, num_classes, checkpoint_dir, pretrained)
