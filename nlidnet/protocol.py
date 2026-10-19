import csv
import random
from collections import defaultdict
from pathlib import Path

LIQUIDS = (
    "water",
    "ethanol_20",
    "ethanol_40",
    "ethanol_60",
    "ethanol_80",
    "ethanol_100",
    "DMSO",
    "EtOAc",
    "BnOH",
    "iPrOH",
    "hexane",
    "NaOH_1M",
    "baijiu_alcohol_56v",
    "sake_alcohol_15v",
    "vodka_alcohol_40v",
    "vinegar_9",
    "coconut_water",
    "NaCl_1M",
    "NaCl_2M",
    "glucose_10",
    "NaHCO3_0.5M",
    "glycerol_10",
    "glycerol_20",
    "glycerol_30",
    "glycerol_40",
    "ethanol_10",
    "ethanol_30",
    "ethanol_50",
    "ethanol_70",
    "ethanol_90",
)
BASE_LABELS = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 21, 22, 23, 24)
NOVEL_LABELS = tuple(range(12, 21))
CONCENTRATIONS = {
    0: 0,
    1: 20,
    2: 40,
    3: 60,
    4: 80,
    5: 100,
    25: 10,
    26: 30,
    27: 50,
    28: 70,
    29: 90,
}
STAGES = {"stage1": 2950, "stage2": 3650, "stage3": 4350}
OFFSETS = tuple(range(0, 50, 5))
ROI_BOXES = (
    (535, 385, 635, 470),
    (635, 385, 735, 470),
    (735, 385, 835, 470),
    (535, 475, 635, 555),
    (635, 475, 735, 555),
    (735, 475, 835, 555),
)


def read_manifest(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"trial_id", "split", "label", "prefix"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"Manifest requires columns {sorted(required)}")
        rows = list(reader)
    return validate_manifest_rows(rows)


def validate_manifest_rows(rows):
    seen = set()
    for row in rows:
        if not row["trial_id"] or row["trial_id"] in seen:
            raise ValueError(f"Duplicate or empty trial_id: {row['trial_id']!r}")
        seen.add(row["trial_id"])
        row["label"] = int(row["label"])
        if row["split"] not in {"train", "test"} or not 0 <= row["label"] < len(
            LIQUIDS
        ):
            raise ValueError(f"Invalid trial: {row}")
        if Path(row["prefix"]).name != row["prefix"] or row["prefix"] in {
            ".",
            "..",
            "",
        }:
            raise ValueError("prefix must be a file stem without directory components")
    pairs = [(r["split"], r["prefix"]) for r in rows]
    if len(set(pairs)) != len(pairs):
        raise ValueError("Duplicate output prefix within a split")
    return rows


def support_indices(records, shots, seed):
    if shots < 1:
        raise ValueError("shots must be positive")
    groups = defaultdict(list)
    for i, r in enumerate(records):
        if r["split"] != "train":
            raise ValueError("Support samples must come from the training split")
        groups[r["label"]].append(i)
    result = []
    for label, indices in sorted(groups.items()):
        indices.sort(key=lambda i: records[i]["trial_id"])
        random.Random(seed + label * 1009).shuffle(indices)
        if len(indices) < shots:
            raise ValueError(
                f"label {label}: requested {shots} trials but found {len(indices)}"
            )
        result.extend(indices[:shots])
    return result
