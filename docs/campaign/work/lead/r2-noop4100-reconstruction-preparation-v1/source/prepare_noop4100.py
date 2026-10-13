#!/usr/bin/env python3
"""Reconstruct the complete reviewed 4,227-row no-op frontier.

This integrates the frozen v3 zero-width reconstruction with the reviewed v4
nonzero/CRLF recovery decisions.  Provider inputs contain source and cursor
only.  Semantic failures become named holds; source I/O failures terminate.
"""
from __future__ import annotations
import argparse, collections, hashlib, json, os, shutil, tempfile
from pathlib import Path
import base_reconstruction as base
import recovery_geometry as geometry

SUPPORTED = base.SUPPORTED
RECOVERED = "recoverable_source_backed"

def req(value, message):
    if not value: raise ValueError(message)

def sha(path): return base.sha(Path(path))

def target_free(value):
    forbidden = [k for k in value if "target" in k.lower() or "gold" in k.lower()]
    req(not forbidden, f"prediction target/gold keys:{forbidden}")

def stable_source(path: Path, expected: str) -> bytes:
    # Missing/unreadable/changing files are infrastructure failures, not holds.
    before = path.stat()
    with path.open("rb") as stream: raw = stream.read()
    after = path.stat()
    req((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
        (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
        f"source changed during read:{path}")
    req(hashlib.sha256(raw).hexdigest()==expected, f"source hash:{path}")
    return raw

def utf16_units(text: str) -> int:
    return len(text.encode("utf-16-le")) // 2

def utf16_column_offset(line: str, column: int) -> int:
    req(type(column) is int and column >= 0, "UTF16 column type")
    used=0
    for index,ch in enumerate(line):
        if used==column: return index
        used+=utf16_units(ch)
        req(used<=column,"UTF16 column splits surrogate pair")
    req(used==column,"UTF16 column outside line")
    return len(line)

def position_offset(text: str, position: dict) -> int:
    line=position.get("line"); column=position.get("character")
    req(type(line) is int and line>=0,"line type")
    lines=text.split("\n"); req(line<len(lines),"line outside text")
    return sum(len(x)+1 for x in lines[:line])+utf16_column_offset(lines[line],column)

def project_cursor_and_selection(packet, provenance, raw):
    result=packet["result"]; context=result["context"]; window_text=result["selection_source"]["text"]
    view,_=geometry.normalized_source(raw,provenance["source_window_occurrence_method"])
    window=window_text.encode("utf-8"); req(view.count(window)==1,"window occurrence")
    byte_offset=view.index(window); prefix=view[:byte_offset].decode("utf-8"); full=view.decode("utf-8")
    replacement=context["replacement_range"]; start=replacement["start"]; end=replacement["end"]
    start_local=position_offset(window_text,start); end_local=position_offset(window_text,end); req(end_local>=start_local,"range reversed")
    selected=window_text[start_local:end_local]; region="\n".join(context["region_old"]); req(selected==region,"actual normalized source range differs")
    base_line=prefix.count("\n"); prefix_column=utf16_units(prefix.rsplit("\n",1)[-1])
    cursor={"line":base_line+start["line"],"character":start["character"]+(prefix_column if start["line"]==0 else 0)}
    global_start=position_offset(full,cursor)
    global_end_position={"line":base_line+end["line"],"character":end["character"]+(prefix_column if end["line"]==0 else 0)}
    global_end=position_offset(full,global_end_position)
    req(full[global_start:global_end]==selected,"projected global source range differs")
    return cursor,selected,{"window_starts_at_line_boundary":byte_offset==0 or view[byte_offset-1:byte_offset]==b"\n","window_start_utf16_column":prefix_column}

def recovered_output(packet, provenance, recorded):
    rid=provenance["row_id"]
    source=Path(packet["validation"]["source_path"])
    raw=stable_source(source, provenance["source_path_sha256"])
    req(packet["validation"]["source_sha256"]==hashlib.sha256(raw).hexdigest(),"packet source hash")
    # Recompute the frozen recovery decision.  Do not derive cursor from gold.
    recomputed=geometry.classify(provenance, packet, recorded.get("old_v3_reason"))
    req(recomputed.get("status")==RECOVERED,"recovery no longer valid")
    for key in ("row_id","shard","package_id","group_id","status","source_window","eol","replay"):
        req(recomputed.get(key)==recorded.get(key),f"recovery decision differs:{key}")
    for key in ("kind","range_start","range_end","selected_region_sha256","selected_region_lines","nonzero_selection","cursor_origin"):
        req(recomputed["geometry"].get(key)==recorded["geometry"].get(key),f"recovery geometry differs:{key}")
    cursor,selected,projection=project_cursor_and_selection(packet,provenance,raw)
    req(cursor["line"]==recorded["geometry"]["global_cursor"]["line"],"recorded cursor line differs")
    req(type(cursor.get("line")) is int and cursor["line"]>=0 and type(cursor.get("character")) is int and cursor["character"]>=0,"cursor shape")
    raw_sha=hashlib.sha256(raw).hexdigest(); preedit=raw.decode("utf-8")
    expected_eol=recorded["eol"]["raw"]
    package_root=source.parent.parent
    prediction={"schema":"sepalith.dat10.sourcewalk-noop.prediction_input.v2","row_id":rid,
      "path":provenance["normalized_relative_source_path"],"preedit_text":preedit,
      "preedit_sha256":raw_sha,"cursor":cursor,"document_eol":expected_eol,
      "absolute_document_path":str(source),"workspace_root":str(package_root),"expected_dependencies":[]}
    target_free(prediction)
    sidecar={"schema":"sepalith.dat10.sourcewalk-noop.training_sidecar.v2","row_id":rid,
      "identity":{"row_id":rid,"package_id":provenance["package_id"],"group_id":provenance["group_id"],
        "family":"no_op","source_path":str(source),"source_sha256":raw_sha,
        "normalized_relative_source_path":provenance["normalized_relative_source_path"],
        "window_sha256":recorded["source_window"]["sha256"],"cursor_origin":"unique_source_window_plus_recorded_UTF16_replacement_start",
        "cursor_projection":projection,"selected_region_sha256":hashlib.sha256(selected.encode("utf-8")).hexdigest(),
        "document_eol":expected_eol},"target_operation":"no_op","target_body_lines":[],
      "full_source_reapplication_sha256":raw_sha,"prediction_target_free":True,
      "recorded_unchanged_selection_nonempty":recorded["geometry"]["nonzero_selection"]}
    req(prediction["preedit_text"].encode("utf-8")==raw,"raw source bytes")
    req(sidecar["full_source_reapplication_sha256"]==raw_sha,"no-op replay")
    return prediction,sidecar

def candidate_output(item, packet, provenance, decisions):
    req(item.get("status")==provenance.get("status"),"candidate status differs from pinned ledger")
    rid=provenance["row_id"]; shard=item["shard"]
    if item["status"]!=SUPPORTED:
        return None,None,{"row_id":rid,"shard":shard,"status":"hold","reason":"provenance_not_supported","source_reasons":item.get("reasons",[]),"silent_drop":False}
    if rid in decisions:
        decision=decisions[rid]
        if decision.get("status")!=RECOVERED:
            return None,None,{"row_id":rid,"shard":shard,"status":"hold","reason":decision.get("reason","reviewed_mixed_eol_hold"),"silent_drop":False,"recovery_decision_bound":True}
        try: return (*recovered_output(packet,provenance,decision),None)
        except geometry.RecoveryError as error:
            return None,None,{"row_id":rid,"shard":shard,"status":"hold","reason":f"RecoveryError:{error}","silent_drop":False}
    try:
        prediction,sidecar=base.recover(packet,provenance)
        prediction.pop("selection_target_or_gold_used",None)
        target_free(prediction)
        return prediction,sidecar,None
    except base.RowValidationError as error:
        return None,None,{"row_id":rid,"shard":shard,"status":"hold","reason":f"RowValidationError:{error}","silent_drop":False}

def main():
    p=argparse.ArgumentParser()
    for name in ("coverage","candidate-ids","recovery-ledger"): p.add_argument("--"+name,type=Path,required=True)
    for name in ("coverage-sha256","candidate-ids-sha256","recovery-ledger-sha256"): p.add_argument("--"+name,required=True)
    p.add_argument("--replay-root",type=Path,required=True);p.add_argument("--packet-root",type=Path,required=True);p.add_argument("--output",type=Path,required=True)
    a=p.parse_args(); req(sha(a.coverage)==a.coverage_sha256 and sha(a.candidate_ids)==a.candidate_ids_sha256 and sha(a.recovery_ledger)==a.recovery_ledger_sha256,"input pins")
    coverage=json.loads(a.coverage.read_text()); candidates=base.rows(a.candidate_ids); decisions=base.rows(a.recovery_ledger)
    req(base.is_full_closure(coverage) and len(candidates)==4227,"full41 candidate closure")
    req(len(decisions)==2991 and sum(x.get("status")==RECOVERED for x in decisions.values())==2985,"recovery closure")
    by_shard=collections.defaultdict(dict)
    for rid,item in candidates.items(): by_shard[item["shard"]][rid]=item
    a.output.parent.mkdir(parents=True,exist_ok=True); req(not a.output.exists(),"fresh output")
    tmp=Path(tempfile.mkdtemp(prefix="."+a.output.name+".",dir=a.output.parent)); all_pred=[];holds=[];per=[]
    try:
      receipt_pins={x["shard"]:x["sha256"] for x in coverage["receipts"]}
      for shard in range(41):
       subset=by_shard.get(shard,{})
       receipt=a.replay_root/f"shard-{shard:04d}/receipt.json"; req(sha(receipt)==receipt_pins[shard],"receipt pin")
       rv=json.loads(receipt.read_text()); ledger_path=a.replay_root/f"shard-{shard:04d}/ledger.jsonl"; packet_path=a.packet_root/f"shard-{shard:04d}/structured-materialization-v1/candidate-packets.jsonl"
       req(sha(ledger_path)==rv["outputs"][0]["sha256"] and sha(packet_path)==rv["binding"]["candidate_packets_sha256"],"shard pins")
       ledger=base.rows(ledger_path); packets=base.rows(packet_path); pred=[];side=[];held=[]
       for rid,item in sorted(subset.items()):
        req(rid in ledger and rid in packets,"row join")
        out=candidate_output(item,packets[rid],ledger[rid],decisions)
        if out[2] is None: pred.append(out[0]);side.append(out[1])
        else: held.append(out[2])
       for suffix,values in ((".jsonl",pred),(".sidecar.jsonl",side),(".holds.jsonl",held)): base.atomic_jsonl(tmp/f"shard-{shard:04d}{suffix}",values)
       all_pred+=pred;holds+=held;per.append({"shard":shard,"candidates":len(subset),"prediction_inputs":len(pred),"holds":len(held),"prediction_sha256":sha(tmp/f"shard-{shard:04d}.jsonl"),"sidecar_sha256":sha(tmp/f"shard-{shard:04d}.sidecar.jsonl"),"holds_sha256":sha(tmp/f"shard-{shard:04d}.holds.jsonl")})
      duplicate=base.validate_prediction_set(all_pred); base.atomic_jsonl(tmp/"duplicate-geometries.jsonl",duplicate)
      reasons=collections.Counter(x["reason"].split(":",1)[0] for x in holds)
      req((len(all_pred),len(holds))==(4100,127),"required 4100/127 closure")
      manifest={"schema":"sepalith.dat10.noop4100.reconstruction.v1","status":"complete_review_only","training_admission":False,"provider_execution_authorized":False,
       "candidate_rows":4227,"existing_zero_width":1115,"recovered_nonzero_or_crlf":2985,"prediction_inputs":4100,"holds":127,
       "hold_accounting":{"provenance":121,"reviewed_mixed_eol":6,"observed_reason_classes":dict(reasons)},"full_41_shard_closure":True,
       "prediction_target_free":True,"full_source_reapplication":True,"raw_source_bytes_preserved":True,
       "duplicate_geometry":{"policy":"retain_all_until_provider_prompt_target_dedup","groups":len(duplicate),"rows_in_groups":sum(x["count"] for x in duplicate),"artifact":"duplicate-geometries.jsonl","sha256":sha(tmp/"duplicate-geometries.jsonl")},
       "inputs":{"coverage_sha256":a.coverage_sha256,"candidate_ids_sha256":a.candidate_ids_sha256,"recovery_ledger_sha256":a.recovery_ledger_sha256},"shards":per}
      (tmp/"manifest.json").write_text(json.dumps(manifest,indent=2,sort_keys=True)+"\n"); os.rename(tmp,a.output); print(json.dumps(manifest,sort_keys=True))
    except Exception: shutil.rmtree(tmp,ignore_errors=True); raise

if __name__=="__main__": main()
