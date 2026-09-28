#!/usr/bin/env python3
"""Cross-check TypeScript serving output with the exact v2 geometry helper."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
HELPER=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign/work/lead/r2-sourcewalk-semantic-materialization-v2/semantic_context_geometry.py')
HELPER_SHA='f38c3a96b5a8b34b420b289c00cf5de3f0c4fb51fba09ea3e634a61299339b6d'
def sha_bytes(x:bytes):return hashlib.sha256(x).hexdigest()
if sha_bytes(HELPER.read_bytes())!=HELPER_SHA:raise SystemExit('materializer_v2_geometry_helper_pin_mismatch')
spec=importlib.util.spec_from_file_location('materializer_v2_geometry_exact',HELPER);g=importlib.util.module_from_spec(spec);sys.modules['materializer_v2_geometry_exact']=g;spec.loader.exec_module(g)
ap=argparse.ArgumentParser();ap.add_argument('input',type=Path);ap.add_argument('output',type=Path);a=ap.parse_args();data=json.loads(a.input.read_text());checks=[]
full=data['lf_full'];cropped=data['lf_cropped'];crlf=data['crlf_cropped']
full_text=g.context_buffer_text(full['context']);assert full_text==full['before'];checks.append('full_context_reconstructs_full_active_document')
assert g.apply_zero_width(full['context'],["#' @title Target docs","#' @param x input"])==full['after'];checks.append('full_context_application_exact')
for label,row in [('lf_cropped',cropped),('crlf_cropped',crlf)]:
 try:g.context_buffer_text(row['context'])
 except ValueError as e:
  assert str(e) in {'expected_zero_width_blank_anchor','context_content_sha256_mismatch'}
  checks.append(label+'_correctly_rejected_by_v2_local_geometry:'+str(e))
 else:raise AssertionError(label+'_unexpected_v2_geometry_acceptance')
out={'schema':'sepalith.run06.materializer_v2_geometry_crosscheck.v1','status':'pass_with_cropped_incompatibility_confirmed','helper_path':str(HELPER),'helper_sha256':HELPER_SHA,'input_sha256':sha_bytes(a.input.read_bytes()),'checks':checks,'checks_passed':len(checks),'interpretation':'Current serving full-document geometry passes v2 only when the full source is retained. Correct bounded serving contexts retain the full active-document identity and global range, so the v2 cropped-view-equals-document-hash condition rejects them.','training_or_serving_admission':False}
a.output.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':out['status'],'checks':len(checks)}))
