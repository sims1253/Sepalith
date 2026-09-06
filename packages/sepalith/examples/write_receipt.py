"""A harmless runner example: write a receipt, not an experiment result."""
import json
from pathlib import Path
import sys


output = Path(sys.argv[1])
output.write_text(json.dumps({"kind": "runner-demo", "message": "step completed"}) + "\n")
print(f"Wrote demo receipt to {output}")
