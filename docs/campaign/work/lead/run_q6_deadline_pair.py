"""Run four serial notebook probes with real deadlines and durable receipts."""
from pathlib import Path
import datetime as dt
import hashlib
import json
import shlex
import subprocess

P = Path('/home/m0hawk/.t3/worktrees/Sepalith/t3code-a8153bbb')
C = P/'docs/campaign'
R = '/home/m0hawk/.local/share/sepalith-campaign-20260915'
HOST = 'm0hawk@192.168.178.40'
SSH = ['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','ServerAliveInterval=15','-o','ServerAliveCountMax=3',HOST]
out = C/'work/lead/notebook-q6-deadline-results'
out.mkdir(exist_ok=True)
helper_sha = '2b087c68b23f367b6b7a8616cd618c3c26115f181a9023d835db2a8e4cf5c0c9'
arms = [('Q8_0',5000),('Q6_K',5000)]
results = []
for candidate, ms in arms:
    run = f'RUN-09-primary500-{candidate.lower()}-latency-{ms}-b'
    check = ("from pathlib import Path\nimport hashlib,socket,json\n"
             f"p=Path('{R}/probe-v2/notebook_quant_latency_probe_v2.py')\n"
             f"assert hashlib.sha256(p.read_bytes()).hexdigest()=='{helper_sha}'\n"
             "s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',18403));s.close()\n")
    subprocess.run(SSH+['python3','-'],input=check,text=True,capture_output=True,timeout=15,check=True)
    command = ['timeout','--signal=TERM','--kill-after=15s','1250s','python3',
               f'{R}/probe-v2/notebook_quant_latency_probe_v2.py', '--backend','vulkan',
               '--build-receipt',f'{R}/runs/backend-build-d/vulkan-terminal.json',
               '--build-receipt-sha256','6f4b5c3064e74cd25013e142417aa6cc71aa66a7591d442a978d74f5a75ba7ed',
               '--candidate',candidate,'--deadline-ms',str(ms),'--no-backend-sampling','--run-name',run]
    if not (out/run/'terminal.json').exists():
        lease = {'task':'RUN-09','owner':'lead','at':dt.datetime.now(dt.timezone.utc).isoformat(),
                 'run':run,'candidate':candidate,'deadline_ms':ms,'remote_command':command,
                 'helper_sha256':helper_sha,'status':'admitted','terminal_pending':True}
        (out/'current-launch.json').write_text(json.dumps(lease,indent=2)+'\n')
        lines=(C/'RESOURCE-LEASES.md').read_text().splitlines()
        lines=[f'| Notebook Vulkan | lead | RUN-09 {run} | {lease["at"]} | One serial arm; outer1250s/server1100s/client650s; no localCUDA |' if s.startswith('| Notebook Vulkan |') else s for s in lines]
        (C/'RESOURCE-LEASES.md').write_text('\n'.join(lines)+'\n')
        print(json.dumps({'launch':run,'at':lease['at']}),flush=True)
        with (out/f'{run}.ssh.log').open('xb') as log:
            proc=subprocess.Popen(SSH+[shlex.join(command)],stdout=log,stderr=subprocess.STDOUT)
            lease['local_ssh_pid']=proc.pid
            (out/'current-launch.json').write_text(json.dumps(lease,indent=2)+'\n')
            code=proc.wait(timeout=1280)
        subprocess.run(['scp','-q','-r',f'{HOST}:{R}/runs/{run}',str(out)],timeout=45,check=True)
    else:
        code=0  # Reuse only a previously completed, hash-verified owned arm below.
    local=out/run
    terminal=json.loads((local/'terminal.json').read_text())
    launch=json.loads((local/'launch.json').read_text())
    assert launch['supervisor_sha256']==helper_sha
    assert terminal['probes'][0]['sha256']==hashlib.sha256((local/f'probe-{ms}ms.json').read_bytes()).hexdigest()
    assert code==0 and terminal['server_exit_code']==0, (code,terminal)
    assert len(terminal['probes'])==1 and terminal['probes'][0]['exit_code'] in [0,1]
    proof=json.loads((local/'offload-runtime-proof.json').read_text())
    probe=json.loads((local/f'probe-{ms}ms.json').read_text())
    assert len(probe['requests'])==8 and probe['summary']['user_deadline_ms']==ms
    assert all(row['protocol_status'] in ['accepted','timeout'] for row in probe['requests'])
    processes=[launch['supervisor_pid'],launch['server_timeout_pid']]
    check=("from pathlib import Path\nimport socket\n"+f"assert all(not Path('/proc',str(p)).exists() for p in {processes!r})\n"+
           "s=socket.socket();s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);s.bind(('127.0.0.1',18403));s.close()\n")
    subprocess.run(SSH+['python3','-'],input=check,text=True,capture_output=True,timeout=15,check=True)
    result={'run':run,'candidate':candidate,'deadline_ms':ms,'terminal':terminal,
            'summary':probe['summary'],'offload':proof,'owned_processes_gone':True,
            'probe_sha256':hashlib.sha256((local/f'probe-{ms}ms.json').read_bytes()).hexdigest()}
    results.append(result)
    (out/'results.json').write_text(json.dumps(results,indent=2)+'\n')
    print(json.dumps({'completed':run,'accepted':probe['summary']['accepted_requests'],'requests':8}),flush=True)
(out/'terminal.json').write_text(json.dumps({'at':dt.datetime.now(dt.timezone.utc).isoformat(),'status':'completed','arms':2,'all_owned_processes_gone':True},indent=2)+'\n')
