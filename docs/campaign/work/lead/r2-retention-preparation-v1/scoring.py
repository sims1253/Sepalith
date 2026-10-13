"""Token-complete causal BPB primitives. No model or corpus access on import."""
import math


def windows(token_ids, bos_id, context=2048, overlap=256):
    """Yield (input IDs, first scored target index, count).

    Each document token is a target exactly once. BOS supplies context for
    the first token and contributes no bytes. Later windows retain context.
    The first scored index addresses input IDs, not shifted logits.
    """
    if not 1 <= overlap < context or context < 2:
        raise ValueError('require 1 <= overlap < context')
    ids = [bos_id] + list(token_ids)
    start = 1
    while start < len(ids):
        left = max(0, start - overlap)
        end = min(len(ids), left + context)
        yield ids[left:end], start - left, end - start
        start = end


def summarize(total_nats, scored_tokens, expected_tokens, corpus_bytes):
    if scored_tokens != expected_tokens:
        raise ValueError('not every token was scored exactly once')
    if not math.isfinite(total_nats) or total_nats < 0:
        raise ValueError('invalid loss sum')
    if corpus_bytes <= 0 or scored_tokens <= 0:
        raise ValueError('empty corpus or token stream')
    return {'total_nats': total_nats, 'tokens': scored_tokens,
            'utf8_bytes': corpus_bytes, 'bits_per_byte': total_nats / math.log(2) / corpus_bytes,
            'nats_per_token': total_nats / scored_tokens}


def score_documents(model, tokenizer, documents, context=2048, overlap=256):
    """Score exact supplied document strings with an already-owned model.

    Caller owns freeze admission, source provenance, model identity, CUDA
    lease and memory guard. This function never loads a model or source file.
    """
    import hashlib
    import torch
    import torch.nn.functional as functional
    if type(tokenizer.bos_token_id) is not int or tokenizer.bos_token_id < 0:
        raise ValueError('explicit tokenizer BOS required')
    device = next(model.parameters()).device
    rows = []
    model.eval()
    with torch.inference_mode():
        for document in documents:
            text = document['text']
            if not isinstance(text, str) or not text:
                raise ValueError('nonempty document text required')
            raw = text.encode('utf-8')
            ids = tokenizer.encode(text, add_special_tokens=False)
            if any(type(token) is not int or token < 0 for token in ids):
                raise ValueError('invalid tokenizer IDs')
            if tokenizer.decode(ids, skip_special_tokens=False,
                                clean_up_tokenization_spaces=False) != text:
                raise ValueError('tokenizer does not preserve exact source bytes')
            total, count = 0.0, 0
            for sequence, first, targets in windows(ids, tokenizer.bos_token_id, context, overlap):
                inputs = torch.tensor([sequence], device=device)
                output = model(input_ids=inputs, use_cache=False).logits
                if output.ndim != 3 or tuple(output.shape[:2]) != (1, len(sequence)):
                    raise ValueError('unexpected causal logits shape')
                if max(sequence) >= output.shape[-1]:
                    raise ValueError('tokenizer ID exceeds model vocabulary')
                logits = output[0]
                for start in range(first, len(sequence), 128):
                    stop = min(start + 128, len(sequence))
                    loss = functional.cross_entropy(logits[start-1:stop-1].float(),
                                                     inputs[0, start:stop], reduction='sum')
                    total += loss.item()
                count += targets
                del logits, output, inputs
            row = summarize(total, count, len(ids), len(raw))
            row.update(id=document['id'], sha256=hashlib.sha256(raw).hexdigest(),
                       tokenizer_roundtrip_exact=tokenizer.decode(ids, skip_special_tokens=False,
                           clean_up_tokenization_spaces=False) == text)
            rows.append(row)
    result = summarize(sum(r['total_nats'] for r in rows), sum(r['tokens'] for r in rows),
                       sum(r['tokens'] for r in rows), sum(r['utf8_bytes'] for r in rows))
    result.update(documents=rows, context=context, overlap=overlap,
                  bos_scored=False, every_document_token_scored_once=True)
    return result
