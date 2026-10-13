import importlib.util,json,hashlib,datetime
from pathlib import Path
P=Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb/docs/campaign')
W=P/'work/lead/theta0-cuda-dev512-a'
f=P/'work/theta0-cuda-dev192-review-v1/review_dev192.py'
s=importlib.util.spec_from_file_location('review',f);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
protocol=m.load_protocol();tok,ta=m.load_tokenizer();panel,pa=m.load_panel(protocol,tok)
r=m.review_run(protocol,tok,panel,Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/training/RUN-05-theta0-cuda-dev512-a/Q8_0-b512'),192,'b512')
assert r['all_rows_independently_checked'] and r['all_saved_decisions_match'] and r['source_checks']['all_checks_pass'] and r['terminal_checks']['clean_exit']
r.update(at=datetime.datetime.now(datetime.timezone.utc).isoformat(),review_source_sha256=hashlib.sha256(f.read_bytes()).hexdigest(),tokenizer_audit=ta,panel_audit=pa,decision='reject_b512_promotion_keep_b256',reason='Same26/43edits but23/32strictnoops and6falsepositives versus accepted25625/32and5;7caps versus6.')
(W/'root-review.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'denominators':r['independent_denominators'],'source_checks':r['source_checks']['all_checks_pass'],'diagnostics':r['server_diagnostics'],'report_sha256':hashlib.sha256((W/'root-review.json').read_bytes()).hexdigest()},indent=2))
