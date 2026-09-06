"""CPU preflight over frozen synthetic inputs and the pinned decoder tokenizer."""
import random

from .data import digest, validate_dataset


def training_order(count, recipe):
    rng = random.Random(recipe['seed'])
    order = []
    limit = recipe['steps_per_arm'] * recipe['effective_batch']
    while len(order) < limit:
        epoch = list(range(count))
        rng.shuffle(epoch)
        order.extend(epoch)
    return order[:limit]


def audit_dataset(dataset, tokenizer, recipe):
    validate_dataset(dataset)
    maxima = dict(encoder_bytes=0, query_tokens=0, retrieval_tokens=0, target_tokens=0)
    counts = {}
    for split in ['train', 'eval']:
        for row in dataset[split]:
            source = '\n'.join(m['path'] + '\n' + m['source'] for m in row['view']['modules'])
            count = {'encoder_bytes': len(source.encode()),
                     'query_tokens': len(tokenizer.encode(row['view']['query'], add_special_tokens=False)),
                     'retrieval_tokens': len(tokenizer.encode(source + '\n', add_special_tokens=False)),
                     'target_tokens': len(tokenizer.encode(row['target'], add_special_tokens=False)) + 1}
            counts[row['id']] = count
            for key in maxima:
                maxima[key] = max(maxima[key], count[key])
    if (maxima['encoder_bytes'] > recipe['encoder']['max_bytes'] or
            maxima['query_tokens'] > recipe['max_prompt_tokens'] or
            maxima['retrieval_tokens'] > recipe['max_retrieval_tokens'] or
            maxima['target_tokens'] > recipe['max_new_tokens']):
        raise ValueError(f'Frozen inputs exceed budgets: {maxima}')
    order = [dataset['train'][i]['id'] for i in training_order(len(dataset['train']), recipe)]
    scored = sum(counts[row]['target_tokens'] for row in order)
    if scored * len(recipe['arms']) > recipe['max_scored_tokens']:
        raise ValueError('Target exposure exceeds the total gate cap')
    return {'maxima': maxima, 'order_sha256': digest(order), 'scored_tokens_per_arm': scored,
            'examples_per_arm': len(order), 'split_sha256': digest(dataset['partitions']),
            'tokenizer_training': 'none; encoder uses fixed UTF-8 byte IDs, decoder tokenizer frozen',
            'train_packages': len(dataset['train']), 'eval_packages': len(dataset['eval'])}
