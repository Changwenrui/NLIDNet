import argparse
import importlib
import sys


COMMANDS = {
    "data": (
        "prepare_data",
        "RAW preparation, trial indexing and event representations",
    ),
    "train": ("train", "Source-category representation learning"),
    "evaluate": ("evaluate", "Trial-level source classification"),
    "adapt": ("adapt", "Few-shot classification and concentration regression"),
    "baselines": ("baselines", "Event, video, temporal and RGB appearance comparisons"),
    "ablations": ("ablations", "Architecture and representation-loss controls"),
    "analyze": (
        "analyze",
        "Physical correlation, features, event statistics and timing",
    ),
    "visualize": ("visualize", "Attention, Grad-CAM, embeddings and videos"),
}


def main(argv=None):
    arguments = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(
        prog="nlidnet",
        description="Active hydrodynamic interrogation with event-based liquid sensing",
    )
    parser.add_argument("command", choices=list(COMMANDS), nargs="?")
    parser.epilog = "\n".join(
        f"{name:12s} {description}" for name, (_, description) in COMMANDS.items()
    )
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    if not arguments or arguments[0] in {"-h", "--help"}:
        parser.print_help()
        return
    aliases = {
        "preprocess": ("data", ["prepare"]),
        "check-data": ("data", ["validate"]),
        "export": ("visualize", ["features"]),
    }
    command = arguments.pop(0)
    if command in aliases:
        command, prefix = aliases[command]
        arguments = prefix + arguments
    if command not in COMMANDS:
        parser.error(f"Unknown command: {command}")
    module = importlib.import_module("nlidnet.commands." + COMMANDS[command][0])
    return module.main(arguments)
