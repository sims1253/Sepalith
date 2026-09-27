"""Build private launch capsules. No launch, network, model or key access."""
from pathlib import Path
import ast, hashlib, json, shutil, tarfile, uuid
from datetime import datetime, timezone
HERE=Path(__file__).resolve().parent
CAMPAIGN=HERE.parents[1]
MERGED=CAMPAIGN/'work/remote-auto-merged-v1'
LEAD=CAMPAIGN/'work/lead'
REMOTE=Path('/home/m0hawk/.local/share/sepalith-campaign-20260915')
MERGED_SHA='c2a658e1789d7ff99d428c70024d0655a664f05164b178e04ca5b7f86c1cecc9'
RECEIPT_SHA='0904404ff9fdfb1081ca23af926fb1ea37f969b1b637a5bc595c578118b62170'
RENDERER_SHA='e04cb8ec68016ccf4557690a2c8c908c3534614af1abe4c5c557157548169d78'
VSIX_SHA='b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1'
EXPECTED={1500:('d9d73e96-1510-4aac-8ff8-57340f9c40a2','51feb0141ff163de8eea7444cad4a006aee4e93a794d5ec2bca06eb42cfa763f'),350:('f3882149-bf3a-4b02-ba01-ae15fed73165','58174b4b3e75764553c545944d111da00b1bb40fd1ed0f0d85ef3ae56d1035b3')}
def pin(p):
    data=p.read_bytes();return {'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}
def write(p,v):p.write_text(json.dumps(v,indent=2)+'\n')
def main():
    assert pin(MERGED/'capsule-manifest.json')['sha256']==MERGED_SHA
    receipt=CAMPAIGN/'receipts/RUN-04-remote-auto-merged-v1.json';assert pin(receipt)['sha256']==RECEIPT_SHA
    merged=json.loads((MERGED/'capsule-manifest.json').read_text());assert len(merged)==10
    for name,digest in merged.items():assert pin(MERGED/name)['sha256']==digest
    primary=json.loads((LEAD/'remote-primary-a/binding.json').read_text())
    original_guard=LEAD/'remote-primary-a/notebook-capsule/notebook_editor_supervisor.py'
    source=original_guard.read_text();assert RENDERER_SHA in source
    pins=[pin(MERGED/'capsule-manifest.json'),pin(receipt),pin(original_guard),pin(LEAD/'remote-primary-a/run_remote_editor.py')]
    plans=[]
    for arm,(instance,binding_sha) in EXPECTED.items():
        name=f'remote-auto{arm}-a';root=LEAD/name;metadata=json.loads((root/'arm.json').read_text());binding=json.loads((root/'binding.json').read_text())
        assert metadata['debounce_ms']==arm and metadata['root_name']==name and metadata['instance_id']==instance
        assert str(uuid.UUID(instance))==instance and pin(root/'binding.json')['sha256']==binding_sha
        assert binding['schema']==1 and binding['endpoint']=='http://127.0.0.1:18403' and binding['backend']=='cuda' and binding['instanceId']==instance
        assert binding['manifest']==primary['manifest'];assert json.loads((root/'primary-manifest.json').read_text())==binding['manifest']
        cuda=binding['manifest']['bundles'];assert len(cuda)==1 and cuda[0]['backend']=='cuda'
        assert cuda[0]['launchProfile']['contextSize']==4096 and cuda[0]['launchProfile']['parallel']==1 and cuda[0]['launchProfile']['cudaGraphOptimization']==0
        assert binding['manifest']['modelProfile']['maxOutputTokens']==192
        assert pin(root/'candidate.vsix')['sha256']==VSIX_SHA and pin(root/'candidate.vsix')['bytes']==40832
        controller=(root/'run_remote_editor.py').read_text();assert f"REMOTE + '/{name}-capsule/notebook_editor_supervisor.py'" in controller
        out=HERE/name;out.mkdir(exist_ok=False)
        for n in merged:
            p=out/n;p.parent.mkdir(exist_ok=True);shutil.copyfile(MERGED/n,p)
        shutil.copyfile(MERGED/'capsule-manifest.json',out/'merged-source-manifest.json')
        for n in ['candidate.vsix','binding.json']:shutil.copyfile(root/n,out/n)
        identity=json.loads((out/'root-supplied-identities.json').read_text());identity.update(binding_source_path=str(root/'binding.json'),binding_sha256=binding_sha,instance_id=instance,debounce_ms=arm,derived_from_merged_manifest_sha256=MERGED_SHA)
        write(out/'root-supplied-identities.json',identity)
        packet=REMOTE/(name+'-capsule');run=REMOTE/'runs'/name;guard=REMOTE/'runs'/(name+'-supervision')
        g=source.replace("'remote-editor-a-capsule'",repr(name+'-capsule')).replace("'runs/remote-editor-a'",repr('runs/'+name)).replace("'runs/remote-editor-a-supervision'",repr('runs/'+name+'-supervision')).replace('26cff670485b2384758af0c478e78aa040e5f1f36d10e3c753257011dd55ce94',binding_sha).replace('223bdcf2-c958-4477-93e2-72a80c769152',instance)
        g=g.replace("'--run-root',str(RUN),'--timeout-ms','180000']",f"'--run-root',str(RUN),'--timeout-ms','180000','--debounce-ms','{arm}']")
        ast.parse(g);(out/'notebook_editor_supervisor.py').write_text(g)
        tree=ast.parse(g);argv_node=next(n.value for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='argv' for t in n.targets))
        argv=eval(compile(ast.Expression(argv_node),'<guard-argv>','eval'),{'__builtins__':{},'str':str},{'PACKET':packet,'RUN':run})
        assert argv[argv.index('--debounce-ms')+1]==str(arm)
        (out/'README.md').write_text(f'# Root-staged {name}\n\nThis capsule derives from merged manifest `{MERGED_SHA}`. Runtime JS files are unchanged.\n\nRun only through root controller `{root}/run_remote_editor.py` after root admission.\nGuard: `{packet}/notebook_editor_supervisor.py`; output `{run}`; supervision `{guard}`.\nArm {arm} ms; instance `{instance}`; binding SHA `{binding_sha}`.\n\nThe historical primary UUID in the merged README is superseded by this exact arm metadata.\nNo launch or native admission follows from staging. Actual visible, multiline and delayed-cancel controls, screenshots and native request evidence remain mandatory.\n')
        files={str(p.relative_to(out)):pin(p)['sha256'] for p in sorted(out.rglob('*')) if p.is_file()};assert 'asset-key.pem' not in files
        write(out/'capsule-manifest.json',files)
        archive=HERE/(name+'.tar')
        with tarfile.open(archive,'w',format=tarfile.USTAR_FORMAT) as tar:
            for n in sorted([*files,'capsule-manifest.json']):
                p=out/n;info=tarfile.TarInfo(n);info.size=p.stat().st_size;info.mode=0o600;info.mtime=0
                with p.open('rb') as f:tar.addfile(info,f)
        plan={'arm_ms':arm,'name':name,'instance_id':instance,'binding_sha256':binding_sha,'local_root_controller':str(root/'run_remote_editor.py'),'remote_capsule':str(packet),'remote_run':str(run),'remote_guard':str(guard),'remote_guard_argv':['python3',str(packet/'notebook_editor_supervisor.py')],'editor_argv':argv,'host':'m0hawk@192.168.178.40','renderer_path':'/usr/share/code/resources/app/out/vs/workbench/workbench.desktop.main.js','renderer_sha256':RENDERER_SHA,'renderer_identity_scope':'Pinned accepted prior notebook guard; fresh remote source SHA check required before transfer/launch.','guard_wall_seconds':300,'desktop_controller_wall_seconds':600,'fresh_native_process_per_arm':True,'capsule_manifest':pin(out/'capsule-manifest.json'),'archive':pin(archive),'files':files,'status':'staged_not_launched_not_admitted'}
        write(HERE/(name+'-plan.json'),plan);plans.append(plan)
        pins.extend(pin(root/n) for n in ['arm.json','binding.json','candidate.vsix','primary-manifest.json','run_remote_editor.py','vsix-identity.json'])
    write(HERE/'staging-plan.json',{'schema':1,'at':datetime.now(timezone.utc).isoformat(),'arms':plans,'input_pins':pins,'private_asset_key_read_or_copied':False,'model_bytes_read':False,'network_or_launch':False})
    print(json.dumps({'status':'prepared','arms':[p['name'] for p in plans],'files_per_arm':[len(p['files'])+1 for p in plans]}))
if __name__=='__main__':main()
