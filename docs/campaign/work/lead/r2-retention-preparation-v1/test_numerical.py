"""CPU toy-model test of shifted causal loss and UTF8 normalization."""
import math
from types import SimpleNamespace
import torch
from scoring import score_documents


class Tokenizer:
    bos_token_id = 3
    alphabet = 'ÄBCD'
    def encode(self, text, add_special_tokens=False):
        assert not add_special_tokens
        return [self.alphabet.index(c) for c in text]
    def decode(self, ids, **kwargs):
        return ''.join(self.alphabet[i] for i in ids)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
    def forward(self, input_ids, use_cache=False):
        assert use_cache is False
        logits = torch.zeros((*input_ids.shape, 4), device=input_ids.device)
        logits.scatter_(-1, ((input_ids + 1) % 4).unsqueeze(-1), 2.0)
        return SimpleNamespace(logits=logits)


torch.set_num_threads(1)
documents = [{'id':'a','text':'ÄBCDÄBCDÄ'}, {'id':'b','text':'Ä'}]
result = score_documents(Toy(), Tokenizer(), documents, context=5, overlap=2)
expected_nats = 10 * (math.log(math.exp(2) + 3) - 2)
assert result['tokens'] == 10
assert result['utf8_bytes'] == 14
assert abs(result['total_nats'] - expected_nats) < 1e-5
assert abs(result['bits_per_byte'] - expected_nats / math.log(2) / 14) < 1e-6
assert all(row['tokenizer_roundtrip_exact'] for row in result['documents'])
print('PASS: exact known causal loss, first token, overlapping windows, final tail, UTF8 BPB')

class LossyTokenizer(Tokenizer):
    def decode(self, ids, **kwargs):
        return 'changed'

class BadShape(Toy):
    def forward(self, input_ids, use_cache=False):
        return SimpleNamespace(logits=torch.zeros(1, 4))

for model, tokenizer, message in [(Toy(), LossyTokenizer(), 'exact source bytes'),
                                  (BadShape(), Tokenizer(), 'logits shape')]:
    try:
        score_documents(model, tokenizer, documents, context=5, overlap=2)
    except ValueError as error:
        assert message in str(error)
    else:
        raise AssertionError('invalid scoring input accepted')
print('PASS: lossy tokenization and invalid model logits rejected')
