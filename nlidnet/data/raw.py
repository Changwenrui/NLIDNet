from pathlib import Path
import numpy as np


def validate_events(events):
    events = np.asarray(events)
    if events.ndim != 2 or events.shape[1] != 4:
        raise ValueError("Events must have columns x, y, t (microseconds), p")
    if events.dtype.kind not in "iuf" or not np.isfinite(events).all():
        raise ValueError("Event coordinates and timestamps must be finite numbers")
    if events.dtype.kind == "f" and not np.equal(events, np.floor(events)).all():
        raise ValueError(
            "Event coordinates and microsecond timestamps must be integers"
        )
    if len(events):
        if np.any(events[1:, 2] < events[:-1, 2]):
            raise ValueError("Event timestamps must be sorted")
        if not np.isin(events[:, 3], (0, 1)).all():
            raise ValueError("Event polarity must be 0 or 1")
        if np.any(events[:, :3] < 0):
            raise ValueError("Event coordinates and timestamps must be nonnegative")
    return np.ascontiguousarray(events, dtype=np.int64)


def read_events(path):
    path = Path(path)
    if path.suffix.lower() == ".raw":
        try:
            from metavision_core.event_io import EventsIterator
        except ImportError as exc:
            raise ImportError(
                "Decoding .raw recordings requires the Prophesee Metavision SDK"
            ) from exc
        chunks = []
        for ev in EventsIterator(input_path=str(path), delta_t=1_000_000):
            if len(ev):
                chunks.append(np.column_stack([ev["x"], ev["y"], ev["t"], ev["p"]]))
        events = np.concatenate(chunks) if chunks else np.empty((0, 4), dtype=np.int64)
    elif path.suffix.lower() == ".safetensors":
        from safetensors.numpy import load_file

        events = load_file(str(path))["events"]
    elif path.suffix.lower() == ".npy":
        events = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=False) as data:
            events = data["events"]
    else:
        raise ValueError(f"Unsupported event file: {path.suffix}")
    return validate_events(events)


def crop_events(events, start_us=0, duration_us=None, roi=None, relative=False):
    events = validate_events(events)
    if start_us < 0 or (duration_us is not None and duration_us <= 0):
        raise ValueError(
            "start_us must be nonnegative and duration_us must be positive"
        )
    if not len(events):
        return events.copy()
    origin = int(events[0, 2]) + int(start_us)
    begin = np.searchsorted(events[:, 2], origin, side="left")
    end = (
        len(events)
        if duration_us is None
        else np.searchsorted(events[:, 2], origin + duration_us, side="left")
    )
    selected = events[begin:end].copy()
    if roi is not None:
        x0, y0, x1, y1 = roi
        if x1 <= x0 or y1 <= y0:
            raise ValueError("ROI must have positive width and height")
        keep = (
            (selected[:, 0] >= x0)
            & (selected[:, 0] < x1)
            & (selected[:, 1] >= y0)
            & (selected[:, 1] < y1)
        )
        selected = selected[keep]
        if relative:
            selected[:, 0] -= x0
            selected[:, 1] -= y0
    if relative:
        selected[:, 2] -= origin
    return selected
