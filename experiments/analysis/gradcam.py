import sys
from nlidnet.cli import main

if __name__ == "__main__":
    main(["visualize", "--kind", "gradcam", *sys.argv[1:]])
