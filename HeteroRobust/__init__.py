import json
import os
from pathlib import Path
import sys

packageRoot = Path(__file__).parent
if (packageRoot / "__environ__.json").exists():
    print("Loading environs...", file=sys.stderr)
    with open(str(packageRoot / "__environ__.json"), "r") as f:
        environDict = json.load(f)
    for key, val in environDict.items():
        os.environ[key] = str(val)
