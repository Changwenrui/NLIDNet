import sys
from nlidnet.cli import main

if __name__ == "__main__":
    main(["check-data", *sys.argv[1:]])
