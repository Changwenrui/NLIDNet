import sys
from nlidnet.cli import main

if __name__ == "__main__":
    main(["preprocess", *sys.argv[1:]])
