from pathlib import Path
import numpy as np
from nlidnet.data.preprocessing import roi_counts
from nlidnet.data.raw import crop_events, validate_events
from nlidnet.data.representations import event_frame
from nlidnet.protocol import ROI_BOXES


def polarity_image(events, roi=(0, 0, 1280, 720)):
    counts = event_frame(events, roi)
    scale = max(float(np.percentile(counts[counts > 0], 98)), 1) if counts.any() else 1
    strength = np.clip(counts / scale, 0, 1)
    image = np.full((counts.shape[1], counts.shape[2], 3), 255, dtype=np.float32)
    positive, negative = strength[1], strength[0]
    image[..., 0] *= 1 - negative
    image[..., 1] *= 1 - np.maximum(positive, negative)
    image[..., 2] *= 1 - positive
    return image.astype(np.uint8)


def preview_recording(
    events,
    output,
    start_bin=2950,
    window_bins=700,
    delta_t_us=2000,
    frame_bins=10,
    max_frames=100,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    events = validate_events(events)
    if not len(events):
        raise ValueError("Cannot preview an empty recording")
    if min(window_bins, delta_t_us, frame_bins, max_frames) < 1 or start_bin < 0:
        raise ValueError("Invalid preview window")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    total_bins = max(9000, int((events[-1, 2] - events[0, 2]) // delta_t_us) + 1)
    counts = roi_counts(events, delta_t_us, total_bins)
    times = np.arange(total_bins) * delta_t_us / 1e6
    fig, axes = plt.subplots(6, 1, figsize=(10, 8), sharex=True)
    for index, ax in enumerate(axes):
        ax.plot(times, counts[index], linewidth=0.6, color="#405b87")
        ax.axvspan(
            start_bin * delta_t_us / 1e6,
            (start_bin + window_bins) * delta_t_us / 1e6,
            alpha=0.12,
        )
        ax.set_ylabel(f"ROI {index + 1}")
    axes[-1].set_xlabel("Time from first event (s)")
    fig.tight_layout()
    fig.savefig(out / "roi_signals.png", dpi=180)
    plt.close(fig)
    np.save(out / "roi_counts.npy", counts, allow_pickle=False)
    window = crop_events(events, start_bin * delta_t_us, window_bins * delta_t_us)
    image = polarity_image(window)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(image)
    for i, (x0, y0, x1, y1) in enumerate(ROI_BOXES):
        ax.add_patch(
            plt.Rectangle(
                (x0, y0), x1 - x0, y1 - y0, fill=False, color="#9f6229", linewidth=1
            )
        )
        ax.text(x0, y0 - 5, str(i + 1), color="#9f6229")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "roi_layout.png", dpi=180)
    plt.close(fig)
    frames = []
    indices = np.arange(start_bin, start_bin + window_bins, frame_bins)
    if len(indices) > max_frames:
        indices = indices[np.linspace(0, len(indices) - 1, max_frames).astype(int)]
    for i, index in enumerate(indices):
        part = crop_events(events, int(index) * delta_t_us, frame_bins * delta_t_us)
        frame = Image.fromarray(polarity_image(part))
        frame.thumbnail((640, 360))
        frame.save(out / f"frame_{i:04d}.png")
        frames.append(frame)
    if frames:
        duration = max(20, int(window_bins * delta_t_us / len(frames) / 1000))
        frames[0].save(
            out / "events.gif",
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0,
        )
    return out
