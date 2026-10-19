import sys
from nlidnet.cli import main

if __name__ == "__main__":
    main(["export", *sys.argv[1:]])
