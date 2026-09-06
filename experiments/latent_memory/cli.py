"""Prepare without GPU access; launch only in a recorded, released resource window."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from .provenance import Ledger, prepare, verify_freeze, write_json


def check_window(path):
    record = json.loads(Path(path).read_text())
    required = ['owner', 'claim_record', 'previous_owner_release', 'quiet_benchmarks_released', 'expires_at']
    if any(not record.get(key) for key in required):
        raise ValueError('Resource-window record lacks ownership/release evidence')
    if record['quiet_benchmarks_released'] is not True or record['expires_at'] <= time.time():
        raise ValueError('Resource window is not released or has expired')
    # Read-only NVML process query; does not create a CUDA context.
    processes = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name',
                                         '--format=csv,noheader'], text=True).strip()
    if processes:
        raise ValueError('GPU has an existing compute workload: ' + processes)
    return record


def worker(frozen, output):
    import torch
    from .experiment import run_screen
    root, output = Path(frozen), Path(output)
    recipe = json.loads((root / 'recipe.json').read_text())
    ledger = Ledger(output, recipe)
    ledger.record('start', python=sys.executable, pid=os.getpid(),
                  affinity=sorted(os.sched_getaffinity(0)))
    try:
        identity = verify_freeze(root)
        torch.set_num_threads(1)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.cuda.set_per_process_memory_fraction(recipe['max_vram_bytes'] / torch.cuda.get_device_properties(0).total_memory)
        torch.cuda.reset_peak_memory_stats()
        ledger.record('hardware', gpu=torch.cuda.get_device_name(), torch_cuda=torch.version.cuda,
                      freeze=identity)
        result = run_screen(root, output, ledger, torch.device('cuda'))
        ledger.record('complete', scientific_status=result['status'], totals=ledger.totals)
    except BaseException as error:
        ledger.record('failure', error=repr(error), totals=ledger.totals)
        write_json(output / 'result.json', {'status': 'INCOMPLETE', 'error': repr(error),
                   'plumbing_success': False, 'learned_information_recovery': 'NOT ESTABLISHED',
                   'editing_improvement': 'NOT TESTED', 'compute': ledger.totals})
        raise


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    prep = commands.add_parser('prepare')
    prep.add_argument('--recipe', default=str(Path(__file__).with_name('recipe.json')))
    prep.add_argument('--checkpoint', required=True)
    prep.add_argument('--output', required=True)
    prep.add_argument('--pin-weights', action='store_true')
    prep.add_argument('--window')
    launch = commands.add_parser('run')
    launch.add_argument('--frozen', required=True)
    launch.add_argument('--output', required=True)
    launch.add_argument('--window', required=True)
    work = commands.add_parser('_worker')
    work.add_argument('--frozen', required=True)
    work.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        if args.pin_weights:
            if not args.window:
                parser.error('--pin-weights requires the resource-window record for checkpoint I/O')
            check_window(args.window)
        freeze = prepare(args.recipe, args.checkpoint, args.output, args.pin_weights)
        print(json.dumps(freeze, indent=2))
    elif args.command == '_worker':
        worker(args.frozen, args.output)
    else:
        window = check_window(args.window)
        root, output = Path(args.frozen).resolve(), Path(args.output).resolve()
        recipe = json.loads((root / 'recipe.json').read_text())
        if window['expires_at'] - time.time() < recipe['max_wall_seconds']:
            raise ValueError('Reserved window is shorter than the frozen wall-clock cap')
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'resource-window.json', window)
        # Foreground child remains in the runner process group. No detached follow-ons.
        started = time.time()
        command = [sys.executable, str(Path(__file__).with_name('entry.py')), '_worker',
                   '--frozen', str(root), '--output', str(output)]
        try:
            subprocess.run(command, check=True, timeout=recipe['max_wall_seconds'])
        except subprocess.TimeoutExpired:
            write_json(output / 'result.json', {'status': 'INCOMPLETE', 'reason': 'hard wall-clock cap',
                       'learned_information_recovery': 'NOT ESTABLISHED', 'editing_improvement': 'NOT TESTED'})
            raise
        finally:
            write_json(output / 'supervisor.json', {'started_at': started, 'ended_at': time.time(),
                       'command': command, 'hard_cap_seconds': recipe['max_wall_seconds']})


if __name__ == '__main__':
    main()
