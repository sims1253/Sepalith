"""Load only the existing pinned R parser packages, not another site's hooks."""
import hashlib
import importlib.util
from pathlib import Path
import sys

ROOT=Path('/home/m0hawk/Documents/Sepalith/.venv/lib/python3.10/site-packages')
PINS={
 'tree_sitter/__init__.py':'0754ef728b311cfb8eb64de361390cbe1344f280c7f40fded08161a6e650268f',
 'tree_sitter/_binding.cpython-310-x86_64-linux-gnu.so':'f900ba170dfc937845fec8994bfb10809f4ad164180f1f1a016a5138aac81212',
 'tree_sitter_r/__init__.py':'9614021d75ff0ccee59f05dfa11aaceade25fe812011c9e3d3fe91a12bed1e2d',
 'tree_sitter_r/_binding.abi3.so':'788d2871015496fcdefa990517ff44dd7d6482c65074a76719c98f205554ff2a',
}

def create_r_parser():
    if sys.version_info[:2]!=(3,10):raise RuntimeError('pinned parser binding requires Python3.10')
    for relative,digest in PINS.items():
        path=ROOT/relative
        if hashlib.sha256(path.read_bytes()).hexdigest()!=digest:raise RuntimeError('R parser source pin mismatch: '+relative)
    for name in ['tree_sitter','tree_sitter_r']:
        expected=ROOT/name/'__init__.py'
        if name in sys.modules:
            if Path(getattr(sys.modules[name],'__file__','')).resolve()!=expected:raise RuntimeError('R parser package origin mismatch')
        else:
            spec=importlib.util.spec_from_file_location(name,expected,submodule_search_locations=[str(expected.parent)])
            module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)
    for name,relative in [('tree_sitter._binding','tree_sitter/_binding.cpython-310-x86_64-linux-gnu.so'),('tree_sitter_r._binding','tree_sitter_r/_binding.abi3.so')]:
        if Path(sys.modules[name].__file__).resolve()!=ROOT/relative:raise RuntimeError('R parser native origin mismatch')
    tree_sitter=sys.modules['tree_sitter'];grammar=sys.modules['tree_sitter_r']
    parser=tree_sitter.Parser(tree_sitter.Language(grammar.language()))
    def parse_r(document):
        if not isinstance(document,str):raise TypeError('R buffer must be text')
        if len(document.encode('utf-8'))>2*1024*1024:raise ValueError('R buffer exceeds candidate byte ceiling')
        return not parser.parse(document.encode('utf-8')).root_node.has_error
    parse_r.identity={'kind':'Tree-sitter syntax only, no R evaluation','pins':dict(PINS),'package_root':str(ROOT)}
    return parse_r
