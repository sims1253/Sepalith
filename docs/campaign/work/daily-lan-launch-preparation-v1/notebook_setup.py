#!/usr/bin/env python3
"""Explicit isolated daily profile setup and fresh-binding receiver. No credentials."""
from pathlib import Path
import argparse,hashlib,json,os,sys,tempfile,urllib.request,uuid
ROOT=Path('/home/m0hawk/.local/share/sepalith-daily-lan')
VSIX_SHA='b8268083aabc72c02623ee07f8b75bb932fd4aaec15e6972724bca9330be74b1'
MANIFEST_SHA='83fa5eb753bd3b58da606406e75e246ffbd13d9ca2965eb9a727c6d52093d791'
def digest(value):return hashlib.sha256(value).hexdigest()
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':')).encode()
def validated_binding(value):
    if set(value)!={'schema','endpoint','instanceId','manifest','backend'} or value['schema']!=1 or value['endpoint']!='http://127.0.0.1:18403' or value['backend']!='cuda':raise ValueError('invalid_binding')
    if str(uuid.UUID(value['instanceId']))!=value['instanceId']:raise ValueError('invalid_instance')
    if digest(canonical(value['manifest']))!=MANIFEST_SHA:raise ValueError('selected_manifest_mismatch')
    return value

def settings():
    return {'sepalith.remoteBindingPath':str(ROOT/'current-binding.json'),'sepalith.manifestUrl':'','sepalith.modelPath':'','sepalith.serverPath':'','sepalith.backend':'cuda','sepalith.port':18403,'sepalith.contextSize':4096,'sepalith.autoStart':False,'sepalith.requestTimeoutMs':5000,'sepalith.debounceMs':1500,'sepalith.scopeContext':False,'sepalith.debugMode':False,'sepalith.postAcceptCooldown':True,'editor.inlineSuggest.enabled':True,'extensions.autoUpdate':False}

def initialize():
    if digest((ROOT/'package/candidate.vsix').read_bytes())!=VSIX_SHA:raise ValueError('selected_VSIX_mismatch')
    target=ROOT/'user-data/User/settings.json';target.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
    data=json.dumps(settings(),indent=2)+'\n'
    if target.exists():
        if target.read_text()!=data:raise ValueError('existing_daily_settings_preserved_review_required')
    else:
        with open(target,'x',opener=lambda p,f:os.open(p,f,0o600)) as f:f.write(data)
    print('Profile prepared; installation and editor launch remain explicit operator actions.')

def receive(raw):
    if len(raw)>1024*1024:raise ValueError('binding_too_large')
    value=validated_binding(json.loads(raw))
    if not (ROOT/'user-data/User/settings.json').exists():raise ValueError('daily_profile_not_initialized')
    if ROOT.is_symlink() or (ROOT/'current-binding.json').is_symlink():raise ValueError('binding_symlink_rejected')
    fd,name=tempfile.mkstemp(prefix='.binding-',dir=ROOT)
    try:
        with os.fdopen(fd,'wb') as f:f.write(json.dumps(value,indent=2).encode()+b'\n');f.flush();os.fsync(f.fileno())
        os.replace(name,ROOT/'current-binding.json')
    finally:
        if Path(name).exists():Path(name).unlink()
    print(json.dumps({'instanceId':value['instanceId'],'bindingSha256':digest((ROOT/'current-binding.json').read_bytes())}))

def probe():
    binding=validated_binding(json.loads((ROOT/'current-binding.json').read_text()))
    def get(route):
        with urllib.request.urlopen(binding['endpoint']+route,timeout=3) as r:
            if r.headers.get('X-Sepalith-Instance-Id')!=binding['instanceId']:raise ValueError('instance_header_mismatch')
            raw=r.read(1024*1024+1)
            if len(raw)>1024*1024:raise ValueError('identity_response_too_large')
            return json.loads(raw)
    m=binding['manifest'];expected={'schema':1,'instanceId':binding['instanceId'],'modelSha256':m['model']['sha256'],'serverSha256':m['bundles'][0]['launchProfile']['serverSha256'],'backend':'cuda','contextSize':4096,'maxOutputTokens':192,'renderer':m['modelProfile']['renderer'],'cudaGraphOptimization':0}
    if get('/sepalith/runtime')!=expected:raise ValueError('runtime_mismatch')
    props=get('/props')
    if props.get('default_generation_settings',{}).get('n_ctx')!=4096 or props.get('total_slots')!=1:raise ValueError('props_mismatch')
    print(json.dumps({'status':'notebook_forward_verified','runtime':expected,'extensionHandshake':'pending actual Sepalith Start server'}))

def main():
    p=argparse.ArgumentParser();p.add_argument('command',choices=['init','bind','probe']);p.add_argument('--self-sha256',required=True);a=p.parse_args()
    if digest(Path(__file__).read_bytes())!=a.self_sha256:raise ValueError('receiver_source_mismatch')
    if a.command=='init':initialize()
    elif a.command=='bind':receive(sys.stdin.buffer.read(1024*1024+1))
    else:probe()
if __name__=='__main__':main()
