"""Pure validation of recorded command rewriting. Does not run build tools."""
from pathlib import Path
import ast
import json
from make_plan import make_plan, parse_command

here = Path(__file__).resolve().parent
metadata = here.parent / 'lead/opus16-build-metadata'
output = Path('/home/m0hawk/.local/share/sepalith-campaign-20260915/runs/opus16-private-native-a')
plan = make_plan(metadata, str(output))
assert plan == json.loads((here / 'build-plan.json').read_text())
assert len(plan['explicit_link_inputs']) == 141
assert plan['reused_shader_objects'] == 138
assert len([x for x in plan['link_argv'] if x.endswith('.o')]) == 139
for flag in ['-o','-MF','-MT','-c']:
    value = plan['compile_argv'][plan['compile_argv'].index(flag)+1]
    assert output in Path(value).parents
assert plan['link_argv'][plan['link_argv'].index('-o')+1] == str(output/'bin/libggml-vulkan.so.0.20.0')
assert '-Wl,--dependency-file='+str(output/'objects/link.d') in plan['link_argv']
assert '-Wl,-rpath,$ORIGIN' in plan['link_argv']
assert not any('glslc' in x or 'vulkan-shaders-gen' in x or x == 'ninja' for x in plan['compile_argv']+plan['link_argv'])
for text in ['/usr/bin/c++ a.o && touch outside', '/usr/bin/c++ @unreviewed.rsp', '/usr/bin/c++ a.cpp\n/usr/bin/c++ b.cpp']:
    try: parse_command(text)
    except ValueError: pass
    else: raise AssertionError('unreviewed command accepted')
for destination in [plan['expected']['protected_build'], plan['expected']['protected_source']+'/elsewhere', 'relative']:
    try: make_plan(metadata,destination)
    except ValueError: pass
    else: raise AssertionError('unsafe output accepted')
for name in ['make_plan.py','collect_fingerprints.py','test_plan.py']:
    ast.parse((here/name).read_text())
print(json.dumps({'status':'PASS','scope':'Recorded argv transformation and refusal rules only; no collector/build/native execution.'}))
