"""Materialize compiler/linker argv only. Never apply, compile, link, or launch."""
from pathlib import Path
import hashlib
import json
import shlex
import sys

HERE = Path(__file__).resolve().parent

def parse_command(text, shell_wrapper=False):
    rows = text.strip().splitlines()
    if len(rows) != 1:
        raise ValueError('Expected exactly one recorded command')
    argv = shlex.split(rows[0])
    if shell_wrapper:
        if argv[:2] != [':', '&&'] or argv[-2:] != ['&&', ':']:
            raise ValueError('Unexpected CMake link wrapper')
        argv = argv[2:-2]
    if any(x in {':', '&&', '||', ';', '|', '>', '<'} or x.startswith('@') for x in argv):
        raise ValueError('Unreviewed shell syntax or response file')
    if argv[0] != '/usr/bin/c++':
        raise ValueError('Compiler differs from reviewed target metadata')
    return argv

def replace_option(argv, flag, expected, replacement):
    if argv.count(flag) != 1 or argv[argv.index(flag) + 1] != expected:
        raise ValueError('Unexpected operand for ' + flag)
    argv[argv.index(flag) + 1] = replacement

def make_plan(metadata, output):
    expected = json.loads((HERE / 'expected-inputs.json').read_text())
    build = Path(expected['protected_build'])
    source = Path(expected['protected_source']) / 'ggml/src/ggml-vulkan/ggml-vulkan.cpp'
    output = Path(output)
    if not output.is_absolute() or output == build or build in output.parents or Path(expected['protected_source']) == output or Path(expected['protected_source']) in output.parents:
        raise ValueError('Output must be an absolute, separate private directory')
    compile_argv = parse_command((metadata / 'compile-commands.txt').read_text())
    link_argv = parse_command((metadata / 'link-commands.txt').read_text(), True)
    host_object = expected['compile_target']
    old_output = expected['link_target']
    object_inputs = [x for x in link_argv if x.endswith('.o')]
    if len(object_inputs) != 139 or object_inputs.count(host_object) != 1:
        raise ValueError('Expected host object plus 138 preserved shader objects')
    replace_option(compile_argv, '-c', str(source), str(output / 'source/ggml-vulkan.cpp'))
    for flag, old, new in [('-o', host_object, 'objects/ggml-vulkan.cpp.o'), ('-MF', host_object + '.d', 'objects/ggml-vulkan.cpp.o.d'), ('-MT', host_object, 'objects/ggml-vulkan.cpp.o')]:
        replace_option(compile_argv, flag, old, str(output / new))
    replace_option(link_argv, '-o', old_output, str(output / 'bin/libggml-vulkan.so.0.20.0'))
    dep = '-Wl,--dependency-file=ggml/src/ggml-vulkan/CMakeFiles/ggml-vulkan.dir/link.d'
    rpath = '-Wl,-rpath,' + str(build / 'bin') + ':'
    if link_argv.count(dep) != 1 or link_argv.count(rpath) != 1:
        raise ValueError('Expected linker dependency output and RPATH were not found')
    link_argv[link_argv.index(dep)] = '-Wl,--dependency-file=' + str(output / 'objects/link.d')
    link_argv[link_argv.index(rpath)] = '-Wl,-rpath,$ORIGIN'
    link_argv[link_argv.index(host_object)] = str(output / 'objects/ggml-vulkan.cpp.o')
    explicit = object_inputs + ['bin/libggml-base.so.0.20.0', '/usr/lib/libvulkan.so']
    return {
        'schema': 'sepalith.opus16.private-native-build-plan.v1',
        'status': 'requires_target_dependency_fingerprints_and_root_review',
        'output_root': str(output), 'cwd': str(build),
        'compile_argv': compile_argv, 'link_argv': link_argv,
        'environment': {'CCACHE_DISABLE': '1', 'TMPDIR': str(output / 'tmp')},
        'unset_environment': ['GGML_VK_Q8_0_M2S', 'GGML_VK_Q8_0_M2S_TRACE'],
        'explicit_link_inputs': [str(build / x) if not Path(x).is_absolute() else x for x in explicit],
        'reused_shader_objects': 138, 'compiled_translation_units': 1,
        'generated_header': str(build / 'ggml/src/ggml-vulkan/ggml-vulkan-shaders.hpp'),
        'protected_metadata': [{'path': str(build / dst), 'sha256': hashlib.sha256((metadata / src).read_bytes()).hexdigest()} for src,dst in [('CMakeCache.txt','CMakeCache.txt'),('build.ninja','build.ninja'),('CMakeFiles_rules.ninja','CMakeFiles/rules.ninja')]],
        'command_metadata_sha256': {n: hashlib.sha256((metadata / n).read_bytes()).hexdigest() for n in ['compile-commands.txt','link-commands.txt']},
        'expected': expected,
        'allowed_write_files': ['source/ggml-vulkan.cpp','objects/ggml-vulkan.cpp.o','objects/ggml-vulkan.cpp.o.d','objects/link.d','bin/libggml-vulkan.so.0.20.0'],
        'commands_execute_shell': False, 'compiler_or_linker_executed': False,
    }

if __name__ == '__main__':
    value = make_plan(Path(sys.argv[1]), sys.argv[2])
    (HERE / 'build-plan.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps({'status': value['status'], 'reused_shader_objects': 138, 'compiled_translation_units': 1}))
