import sys
from pathlib import Path

# Let tests import reward.py and the repo's fwlearn package.
HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent)]
