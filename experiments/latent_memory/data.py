"""Parent-only input views; synthetic package splits precede tokenization."""
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import PurePosixPath
import random
import string


def digest(value):
    if not isinstance(value, bytes):
        value = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class Module:
    path: str
    source: str
    sha256: str
    original_sha256: str
    omitted_ranges: tuple


@dataclass(frozen=True)
class InputView:
    parent_revision: str
    active_path: str
    query: str
    modules: tuple


def prepare_view(parent_files, snapshot_revision, expected_parent, active_path, allowed_paths, masks, query):
    if snapshot_revision != expected_parent:
        raise ValueError('Only the declared parent snapshot may supply evidence')
    if len(set(allowed_paths)) != len(allowed_paths):
        raise ValueError('Duplicate candidates')
    masked_contents = {parent_files[p] for p in masks}
    modules = []
    for path in sorted(allowed_paths):
        if path == active_path or PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts:
            raise ValueError('Invalid or active module candidate')
        source = parent_files[path]
        if source in masked_contents and path not in masks:
            raise ValueError('Unmasked duplicate of a masked source')
        original = digest(source.encode())
        ranges = sorted(masks.get(path, []))
        end = 0
        for start, stop in ranges:
            if not (end <= start < stop <= len(source)):
                raise ValueError('Invalid mask range')
            end = stop
        for start, stop in reversed(ranges):
            source = source[:start] + '[MASKED]' + source[stop:]
        modules.append(Module(path, source, digest(source.encode()), original, tuple(map(tuple, ranges))))
    hidden_spans = [parent_files[p][start:stop] for p, ranges in masks.items() for start, stop in ranges]
    if any(span in module.source for span in hidden_spans for module in modules):
        raise ValueError('Masked span remains in a sibling or generated source')
    if not modules:
        raise ValueError('A source module is required')
    return InputView(expected_parent, active_path, query, tuple(modules))


def make_dataset(seed=1273, train_count=512, eval_count=160):
    # Establish whole repository/package membership BEFORE source generation.
    partitions = {s: [f'{s}-package-{i:04}' for i in range(n)]
                  for s, n in [('train', train_count), ('eval', eval_count)]}
    result = {'schema': 'latent-synthetic-defaults-v1', 'seed': seed, 'partitions': partitions}
    for split, packages in partitions.items():
        rng = random.Random(seed + (0 if split == 'train' else 100000))
        rows = []
        for package in packages:
            names = []
            while len(names) < 4:
                name = 'f_' + ''.join(rng.choices(string.ascii_lowercase, k=6))
                if name not in names:
                    names.append(name)
            defaults = rng.sample(range(10, 100), 4)
            target_index = rng.randrange(4)
            source = '\n'.join(f'{name} <- function(x = {value}) x' for name, value in zip(names, defaults)) + '\n'
            parent = {'R/facts.R': source, 'R/local.R': '# fact query; no answer here\n'}
            tree = digest(parent)
            query = f'What is the default value of x in {names[target_index]}? Return only the integer.\nAnswer:'
            view = prepare_view(parent, tree, tree, 'R/local.R', ['R/facts.R'], {}, query)
            answer = str(defaults[target_index])
            rows.append({'id': package, 'repository': package, 'package': package, 'split': split,
                         'workspace_tree': tree, 'view': asdict(view), 'target': answer,
                         'target_sha256': digest(answer.encode()), 'target_range': None,
                         'analyzer': 'synthetic-definitions-v1', 'candidate_order': ['R/facts.R']})
        result[split] = rows
    validate_dataset(result)
    return result


def validate_dataset(data):
    seen_repositories, seen_packages, seen_sources, seen_ids = set(), set(), set(), set()
    for split in ['train', 'eval']:
        repositories, packages, sources = set(), set(), set()
        for row in data[split]:
            if row['split'] != split or row['package'] not in data['partitions'][split]:
                raise ValueError('Partition mismatch')
            if row['id'] in seen_ids:
                raise ValueError('Duplicate row')
            seen_ids.add(row['id'])
            repositories.add(row['repository'])
            packages.add(row['package'])
            if row['view']['parent_revision'] != row['workspace_tree']:
                raise ValueError('Parent identity mismatch')
            if digest(row['target'].encode()) != row['target_sha256']:
                raise ValueError('Target identity mismatch')
            for module in row['view']['modules']:
                if module['path'] == row['view']['active_path']:
                    raise ValueError('Active file in remote memory')
                if digest(module['source'].encode()) != module['sha256']:
                    raise ValueError('Source identity mismatch')
                sources.add(module['sha256'])
        if repositories & seen_repositories or packages & seen_packages or sources & seen_sources:
            raise ValueError('Repository/package/source overlap across partitions')
        seen_repositories |= repositories
        seen_packages |= packages
        seen_sources |= sources


def input_view(row):
    value = row['view']
    return InputView(value['parent_revision'], value['active_path'], value['query'],
                     tuple(Module(**module) for module in value['modules']))


def git_parent_view(repository, edit_commit, active_path, allowed_paths, masks, query):
    """Read only explicit blobs from edit_commit's first parent, never the worktree.

    The selector must supply a reviewed allowlist excluding generated artifacts and
    unavailable sibling versions. This function does not execute repository code.
    """
    import subprocess
    def git(*args):
        return subprocess.check_output(['git', '-C', str(repository), *args])
    parent = git('rev-parse', '--verify', edit_commit + '^1^{commit}').decode().strip()
    paths = sorted(set(allowed_paths) | set(masks))
    for path in paths:
        if (PurePosixPath(path).is_absolute() or '..' in PurePosixPath(path).parts or
                any(part in {'generated', 'dist', 'build', '.git'} for part in PurePosixPath(path).parts)):
            raise ValueError('Disallowed source path')
    files = {path: git('show', parent + ':' + path).decode('utf-8') for path in paths}
    return prepare_view(files, parent, parent, active_path, allowed_paths, masks, query)
