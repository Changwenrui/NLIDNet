import json
import re
import shutil
import subprocess
from pathlib import Path
import numpy as np


def ffmpeg_executable():
    executable = shutil.which("ffmpeg")
    if executable:
        return executable
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError as exc:
        raise ImportError(
            "Install FFmpeg or pip install -e '.[media]' to export MP4 videos"
        ) from exc


def write_mp4(frames, output, fps=50):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if fps <= 0:
        raise ValueError("fps must be positive")
    iterator = iter(frames)
    try:
        first = np.asarray(next(iterator), dtype=np.uint8)
    except StopIteration as exc:
        raise ValueError("Cannot encode an empty frame sequence") from exc
    if first.ndim != 3 or first.shape[2] != 3:
        raise ValueError("Video frames must be RGB images")
    height, width = first.shape[:2]
    if height % 2 or width % 2:
        raise ValueError("H.264 frames require even width and height")
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg_executable(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "-s",
        f"{width}x{height}",
        "-r",
        str(fps),
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264",
        "-crf",
        "20",
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    count = 0
    try:
        for frame in _including_first(first, iterator):
            array = np.asarray(frame, dtype=np.uint8)
            if array.shape != first.shape:
                raise ValueError("All video frames must have the same dimensions")
            process.stdin.write(np.ascontiguousarray(array).tobytes())
            count += 1
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", errors="replace")
        if process.wait() != 0:
            raise RuntimeError(f"FFmpeg failed: {error}")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        process.stderr.close()
    return {
        "frames": count,
        "fps": fps,
        "duration_seconds": count / fps,
        "width": width,
        "height": height,
    }


def _including_first(first, remaining):
    yield first
    yield from remaining


def font(size):
    from PIL import ImageFont

    for name in ("DejaVuSans.ttf", "C:/Windows/Fonts/arial.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            pass
    return ImageFont.load_default()


def frame_number(path):
    numbers = re.findall(r"\d+", path.stem)
    if not numbers:
        raise ValueError(f"Frame filename has no time index: {path.name}")
    return int(numbers[-1])


def frames_video(source, output, fps=50, delta_t_us=2000, crop=None, accumulate=1):
    from PIL import Image, ImageChops, ImageDraw

    if accumulate < 1 or delta_t_us < 1:
        raise ValueError("accumulate and delta_t_us must be positive")
    paths = sorted(Path(source).glob("frame_*.png"), key=frame_number)
    if not paths:
        raise ValueError(f"No frame_*.png images in {source}")
    indices = [frame_number(p) for p in paths]
    if len(indices) > 1 and len(set(np.diff(indices))) != 1:
        raise ValueError(
            "Frame indices are not regularly sampled; supply a contiguous sequence"
        )
    step_us = (indices[1] - indices[0]) * delta_t_us if len(indices) > 1 else delta_t_us

    def generate():
        for start in range(0, len(paths), accumulate):
            group = paths[start : start + accumulate]
            frame = Image.open(group[0]).convert("RGB")
            for path in group[1:]:
                frame = ImageChops.darker(frame, Image.open(path).convert("RGB"))
            if crop is not None:
                frame = frame.crop(tuple(crop))
            frame.thumbnail((520, 580))
            canvas = Image.new("RGB", (640, 700), "white")
            draw = ImageDraw.Draw(canvas)
            draw.text(
                (32, 18), "Event response during pouring", fill="#252a30", font=font(24)
            )
            time = indices[start] * delta_t_us / 1e6
            draw.text(
                (32, 54), f"Recording time: {time:.3f} s", fill="#5b6670", font=font(18)
            )
            canvas.paste(frame, ((640 - frame.width) // 2, 94))
            draw.text(
                (32, 651),
                "Red: positive events   Blue: negative events",
                fill="#5b6670",
                font=font(17),
            )
            yield canvas

    metadata = write_mp4(generate(), output, fps)
    metadata.update(
        {
            "source": str(source),
            "first_bin": indices[0],
            "last_bin": indices[-1],
            "delta_t_us": delta_t_us,
            "sampling_interval_us": step_us,
            "accumulated_snapshots_per_frame": accumulate,
        }
    )
    Path(output).with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return metadata


def flow_cache_video(source, output, sample_step=10, slowdown=5, gif=None):
    from PIL import Image, ImageDraw
    from matplotlib.colors import hsv_to_rgb

    if sample_step < 1 or slowdown <= 0:
        raise ValueError("sample_step and slowdown must be positive")
    stages = []
    for path in sorted(Path(source).glob("stage*/meta.json")):
        meta = json.loads(path.read_text(encoding="utf-8"))
        counts = np.load(path.parent / "event_count.npy", mmap_mode="r")
        flow = np.load(path.parent / "flow_uv.npy", mmap_mode="r")
        if counts.shape != (flow.shape[0], *flow.shape[-2:]) or flow.shape[1] != 2:
            raise ValueError(f"Mismatched event/flow cache: {path.parent}")
        if len(counts) != meta["num_bins"]:
            raise ValueError(f"Cache length differs from metadata: {path.parent}")
        stages.append((meta, counts, flow))
    if not stages:
        raise ValueError("No stage*/meta.json event/flow caches found")
    if len({meta["delta_ms"] for meta, _, _ in stages}) != 1:
        raise ValueError("All stages must use the same temporal resolution")
    magnitudes, event_levels = [], []
    for _, counts, flow in stages:
        for index in range(0, len(counts), max(50, sample_step)):
            field = np.asarray(flow[index], dtype=np.float32)
            magnitudes.extend(np.hypot(field[0], field[1])[::8, ::8].ravel())
            event_levels.extend(np.asarray(counts[index])[::8, ::8].ravel())
    magnitude_scale = max(float(np.percentile(magnitudes, 98)), 1e-6)
    event_scale = max(float(np.percentile(np.log1p(event_levels), 99)), 1)
    delta_ms = stages[0][0]["delta_ms"]
    fps = 1000 / (delta_ms * sample_step * slowdown)
    gif_frames = []

    def generate():
        for fallback_number, (meta, counts, flow) in enumerate(stages, 1):
            match = re.search(r"stage(\d+)", meta.get("stage", ""))
            stage_number = int(match.group(1)) if match else fallback_number
            for index in range(0, len(counts), sample_step):
                field = np.asarray(flow[index], dtype=np.float32)
                magnitude = np.hypot(field[0], field[1])
                hue = (np.arctan2(field[1], field[0]) / (2 * np.pi) + 1) % 1
                hsv = np.stack(
                    [
                        hue,
                        np.ones_like(hue),
                        np.clip(magnitude / magnitude_scale, 0, 1),
                    ],
                    axis=-1,
                )
                flow_rgb = (hsv_to_rgb(hsv) * 255).astype(np.uint8)
                intensity = np.clip(
                    np.log1p(np.asarray(counts[index], dtype=np.float32)) / event_scale,
                    0,
                    1,
                )
                event_rgb = np.repeat(
                    (intensity * 255).astype(np.uint8)[..., None], 3, axis=-1
                )
                canvas = Image.new("RGB", (960, 680), "white")
                draw = ImageDraw.Draw(canvas)
                draw.text(
                    (34, 18),
                    "Event response and estimated optical flow",
                    fill="#252a30",
                    font=font(26),
                )
                time = (
                    meta["start_sec_relative_to_recording_t0"] + index * delta_ms / 1000
                )
                draw.text(
                    (34, 55),
                    f"Stage {stage_number}  |  Recording time {time:.3f} s  |  {slowdown:g}x slower playback",
                    fill="#5b6670",
                    font=font(18),
                )
                for x, array, title in (
                    (100, event_rgb, "Event count"),
                    (540, flow_rgb, "Frozen EVFlowNet"),
                ):
                    image = Image.fromarray(array)
                    image.thumbnail((320, 480))
                    canvas.paste(image, (x, 130))
                    draw.text((x, 99), title, fill="#252a30", font=font(21))
                draw.text(
                    (34, 633),
                    "Flow hue: direction   |   Brightness: magnitude   |   Shared scale across stages",
                    fill="#5b6670",
                    font=font(17),
                )
                if gif is not None and (index // sample_step) % 3 == 0:
                    thumb = canvas.copy()
                    thumb.thumbnail((640, 454))
                    gif_frames.append(
                        thumb.quantize(colors=64, method=Image.Quantize.MEDIANCUT)
                    )
                yield canvas

    metadata = write_mp4(generate(), output, fps)
    metadata.update(
        {
            "source": str(source),
            "sample_step": sample_step,
            "slowdown": slowdown,
            "flow_magnitude_scale": magnitude_scale,
            "event_log_count_scale": event_scale,
        }
    )
    Path(output).with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    if gif is not None and gif_frames:
        Path(gif).parent.mkdir(parents=True, exist_ok=True)
        gif_frames[0].save(
            gif,
            save_all=True,
            append_images=gif_frames[1:],
            duration=round(3000 / fps),
            loop=0,
            optimize=True,
        )
    return metadata
