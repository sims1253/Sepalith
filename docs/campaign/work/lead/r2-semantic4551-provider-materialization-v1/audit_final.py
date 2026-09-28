#!/usr/bin/env python3
"""Independent closure checks for the review-only 4,551-row preparation."""
import argparse, collections, hashlib, json
from pathlib import Path

PREDICTION_SHA="5bba7f3806bddeb58d11209a852aa397ad1b7f55cb368519d95ac5b7358b19aa"
SIDECAR_SHA="e31cc3c910de6fb426a6bda8c35382cc4dd50514c4ea5fd196513943cd0d1da6"

def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda:stream.read(4<<20),b""):h.update(block)
    return h.hexdigest()

def keyed(path,key="row_id"):
    result={}
    with Path(path).open() as stream:
        for line in stream:
            value=json.loads(line);identity=value[key]
            if identity in result:raise ValueError(f"duplicate:{path}:{identity}")
            result[identity]=value
    return result

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--root",type=Path,required=True);parser.add_argument("--predictions",type=Path,required=True);parser.add_argument("--sidecar",type=Path,required=True);parser.add_argument("--final",type=Path,required=True);args=parser.parse_args()
    assert sha(args.predictions)==PREDICTION_SHA and sha(args.sidecar)==SIDECAR_SHA
    predictions=keyed(args.predictions);sidecar=keyed(args.sidecar);assert len(predictions)==4551 and set(predictions)==set(sidecar)
    selected=keyed(args.root/"selected-01/selected-contexts.jsonl");assert set(selected)==set(predictions)
    for stage in ("render-04","render-16k"):
        for shard in ("0000","0001"):
            terminal=json.loads((args.root/stage/f"shard-{shard}.terminal.json").read_text());output=args.root/stage/f"shard-{shard}.jsonl"
            assert terminal["status"]=="complete" and terminal["output"]["sha256"]==sha(output) and terminal["output"]["rows"]==len(keyed(output))
    manifest=json.loads((args.final/"manifest.json").read_text());assert manifest["status"]=="complete_review_only_root_admission_required" and manifest["denominator"]==4551
    ledger=keyed(args.final/"decision-ledger.jsonl");candidates=keyed(args.final/"candidate-tokenrows.jsonl",key="id");provenance=keyed(args.final/"candidate-provenance.jsonl")
    assert set(ledger)==set(predictions) and set(candidates)==set(provenance)=={row_id for row_id,value in ledger.items() if value["status"]=="candidate"}
    assert len(candidates)+sum(value["status"]!="candidate" for value in ledger.values())==4551
    assert all(value["selection_target_or_gold_used"] is False and value["target_truncated"] is False for value in provenance.values())
    assert all(row["split"]=="train" and row["family"]=="roxygen_drafting" for row in candidates.values())
    assert all(isinstance(token,int) and not isinstance(token,bool) and 0<=token<130560 for row in candidates.values() for token in row["input_ids"])
    assert all(isinstance(token,int) and not isinstance(token,bool) and 0<=token<130560 for row in candidates.values() for key in ("target_body_tokens","target_terminal_tokens") for token in row[key])
    prompt_target=[(hashlib.sha256(row["prompt_text"].encode()).hexdigest(),hashlib.sha256(row["target_body_text"].encode()).hexdigest()) for row in candidates.values()]
    assert len(prompt_target)==len(set(prompt_target))
    counts=collections.Counter(value["reason"] for value in ledger.values())
    assert counts["new_candidate"]==manifest["status_counts"]["new_candidate"]
    assert counts["target_exceeds_generation_reserve"]==manifest["status_counts"]["target_exceeds_generation_reserve"]
    assert sum(n for reason,n in counts.items() if reason not in {"new_candidate","target_exceeds_generation_reserve"})==manifest["status_counts"]["provider_or_policy_hold"]
    print(json.dumps({"status":"PASS","denominator":4551,"candidates":len(candidates),"holds_or_exclusions":4551-len(candidates),"ledger_sha256":sha(args.final/"decision-ledger.jsonl"),"candidate_sha256":sha(args.final/"candidate-tokenrows.jsonl")},sort_keys=True))

if __name__=="__main__":main()
