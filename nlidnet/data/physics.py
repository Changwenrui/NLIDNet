import numpy as np
import torch

PHYS_RAW = {
    0: (998.2, 1.0, 72.8),
    1: (968.6, 2.2, 38.6),
    2: (935.2, 2.9, 30.7),
    3: (891.1, 2.7, 26.7),
    4: (843.4, 2.1, 24.3),
    5: (789.3, 1.2, 22.3),
    6: (1100.4, 2.5, 43.4),
    7: (900.4, 0.5, 23.8),
    8: (1045.0, 5.8, 39.2),
    9: (785.6, 2.4, 21.5),
    10: (659.5, 0.3, 18.5),
    11: (1042.0, 1.1, 72.3),
    17: (1038.0, 1.1, 74.0),
    18: (1073.0, 1.2, 76.0),
    19: (1035.0, 1.4, 73.0),
    20: (1027.0, 1.1, 72.6),
    21: (1024.0, 1.3, 70.5),
    22: (1049.0, 1.8, 71.7),
    23: (1075.0, 2.5, 67.3),
    24: (1101.0, 3.7, 70.0),
    25: (981.9, 1.6, 48.1),
    26: (953.8, 2.6, 33.5),
    27: (913.8, 2.9, 28.5),
    28: (867.7, 2.5, 25.5),
    29: (818.0, 1.7, 23.2),
}

MAIN_ALLOWED_LABELS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24)

_phys_arr = np.array([PHYS_RAW[x] for x in MAIN_ALLOWED_LABELS], dtype=np.float32)
PHYS_MEAN = _phys_arr.mean(axis=0)
PHYS_STD = _phys_arr.std(axis=0) + 1e-6


def get_normalized_phys_target(raw_label: int) -> torch.Tensor:
    if int(raw_label) not in MAIN_ALLOWED_LABELS:
        raise ValueError(
            "Physical auxiliary targets are defined only for the 16 base labels"
        )
    raw = np.array(PHYS_RAW[int(raw_label)], dtype=np.float32)
    norm = (raw - PHYS_MEAN) / PHYS_STD
    return torch.tensor(norm, dtype=torch.float32)
