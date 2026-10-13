"""Small, read-only qualification checks for this pinned ngram editor run."""
import ast
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
MODEL_SHA = 'd269a9fb85cd19efa05c6bf0dc11ccaa0f931fc50b826d893d58ae65043e02db'
MODEL_PATH = '/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT11-task-global-b-500-quant/model-Q8_0.gguf'
SPECULATION = {'type': 'ngram-mod', 'draft_n_max': 64, 'n_match': 24, 'n_min': 48, 'n_max': 64}
FLAGS = {'--spec-type': 'ngram-mod', '--spec-draft-n-max': '64', '--spec-ngram-mod-n-match': '24', '--spec-ngram-mod-n-min': '48', '--spec-ngram-mod-n-max': '64', '-m': MODEL_PATH, '-c': '4096', '-b': '256', '-ub': '256', '--parallel': '1', '--host': '127.0.0.1', '--port': '18403', '-ngl': '99'}

def digest(path):
    # Inputs are bounded source/config/VSIX files, never model artifacts.
    path = Path(path)
    if path.stat().st_size > 2 * 1024 * 1024: raise ValueError('qualification_input_too_large')
    return hashlib.sha256(path.read_bytes()).hexdigest()

def verify_ngram_argv(argv):
    for flag, expected in FLAGS.items():
        if argv.count(flag) != 1 or argv.index(flag) + 1 >= len(argv) or argv[argv.index(flag)+1] != expected:
            raise ValueError('native_execution_argument_mismatch:' + flag)
    if any(x in argv for x in ['-md', '--model-draft', '--draft-model']): raise ValueError('unexpected_draft_model')
    return True

def native_argv_from_pinned_source(package):
    # Evaluate only the reviewed argument-return function and path constants.
    tree = ast.parse((package/'daily_lan.py').read_text())
    constants = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0], ast.Name):
            name = node.targets[0].id
            if name in ['NATIVE', 'BIN', 'MODEL']:
                constants[name] = eval(compile(ast.Expression(node.value), '<pinned-path>', 'eval'), {'Path':Path, **constants})
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name=='native_argv')
    namespace = dict(constants)
    exec(compile(ast.Module(body=[function], type_ignores=[]), '<pinned-native-argv>', 'exec'), namespace)
    return namespace['native_argv']()

def verify_local_inputs():
    pins = json.loads((HERE/'input-pins.json').read_text())
    package, capsule = Path(pins['package']), Path(pins['capsule'])
    if digest(package/'source-manifest.json') != pins['source_manifest_sha256']: raise ValueError('package_manifest_changed')
    for name, expected in pins['package_files'].items():
        if digest(package/name) != expected: raise ValueError('package_source_changed:' + name)
    if digest(capsule/'source-manifest.json') != pins['capsule_manifest_sha256']: raise ValueError('capsule_manifest_changed')
    for name, expected in pins['capsule_runtime_files'].items():
        if digest(capsule/name) != expected: raise ValueError('capsule_source_changed:' + name)
    if digest(pins['guard']) != pins['guard_sha256'] or digest(Path(pins['guard']).with_name('host_memory_policy.py')) != pins['guard_policy_sha256']:
        raise ValueError('host_guard_source_changed')
    profile = json.loads((package/'runtime-execution-profile.json').read_text())
    if profile['status']!='qualification_only' or profile['target_q8_sha256']!=MODEL_SHA or profile['speculation']!=SPECULATION:
        raise ValueError('execution_profile_mismatch')
    argv = native_argv_from_pinned_source(package)
    verify_ngram_argv(argv)
    return pins, argv

def verify_owned_native(session, ident):
    expected = json.loads((session/'native-identity.json').read_text())
    if ident(expected['pid']) != expected: raise ValueError('native_owned_identity_changed')
    argv = (Path('/proc')/str(expected['pid'])/'cmdline').read_bytes().decode().rstrip('\0').split('\0')
    verify_ngram_argv(argv)
    if ident(expected['pid']) != expected: raise ValueError('native_owned_identity_changed')
    return {'identity': expected, 'argv': argv, 'speculation': SPECULATION, 'scope': 'Owned native PID/starttick and actual process argv; native log/task outcomes require post-run review.'}
