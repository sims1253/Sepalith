#!/usr/bin/env python3
"""Run the pinned four-way DEV comparison with the observed E750 hash."""
import hashlib,subprocess,sys
from pathlib import Path
H=Path(__file__).resolve().parent
E=Path('/mnt/e/sepalith/campaign-20260915/training/SFT11-expanded-e750-native-dev-v1/dev-results/results.json')
O=H/'e750-four-way-readout.json'
def main():
 digest=hashlib.sha256(E.read_bytes()).hexdigest()
 return subprocess.run([sys.executable,'-B',str(H/'compare_native_dev.py'),'--e750',str(E),'--e750-sha256',digest,'--output',str(O)],check=False).returncode
if __name__=='__main__':raise SystemExit(main())
