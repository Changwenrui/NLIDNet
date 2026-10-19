import argparse


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features", required=True)
    p.add_argument("--reference-features")
    p.add_argument("--output", default="outputs/embedding_quality.json")
    args = p.parse_args()
    from nlidnet.analysis.embeddings import analyze

    analyze(args.features, args.output, args.reference_features)


if __name__ == "__main__":
    main()
