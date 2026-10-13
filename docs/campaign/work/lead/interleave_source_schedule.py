"""Interleave an admitted finite draw multiset without changing family row order."""
from collections import Counter, defaultdict, deque
import hashlib


def interleave(row_ids, row_families, seed=3407):
    queues = defaultdict(deque)
    for row_id in row_ids:
        queues[row_families[row_id]].append(row_id)
    quotas = {family: len(queue) for family, queue in queues.items()}
    total = len(row_ids)
    used = Counter()
    tie = {family: hashlib.sha256(f'{seed}:interleave-v1:{family}'.encode()).hexdigest()
           for family in quotas}
    result = []
    for i in range(total):
        eligible = [family for family in quotas if queues[family]]
        family = max(eligible, key=lambda f: ((i + 1) * quotas[f] - used[f] * total, tie[f]))
        result.append(queues[family].popleft())
        used[family] += 1
    if Counter(result) != Counter(row_ids) or dict(used) != quotas:
        raise ValueError('Interleaving changed admitted source exposure')
    for family in quotas:
        if [x for x in result if row_families[x] == family] != [x for x in row_ids if row_families[x] == family]:
            raise ValueError('Interleaving changed within-family order')
    return result
