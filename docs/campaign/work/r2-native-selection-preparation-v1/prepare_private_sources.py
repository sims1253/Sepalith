import hashlib,json,shutil
from pathlib import Path
H=Path(__file__).resolve().parent;W=H.parent
# Explicit files only; never inspect models or final data.
E=W/'final-native-evaluator-preparation-v3';C=W/'final-native-rehearsal-preparation-v3'
for root,names,out in [(E,['native_identity.py','native_transport.py','native_evaluator.py','rehearse_dev.py','final_binding.py','final_row_gate.py'],H/'native_evaluator'),(C,['root_controller.py','guarded_dev_client.py','client_source_policy.py','origin_snapshot.py','observe_client.py','build_closure.py','native-mapped-code-allowlist.json','elf-loader-metadata.json'],H/'native_controller')]:
 out.mkdir(exist_ok=True)
 for name in names:shutil.copyfile(root/name,out/name)
for name in ('root_controller.py','guarded_dev_client.py','build_closure.py'):
 p=H/'native_controller'/name;s=p.read_text().replace("HERE.parent/'final-native-evaluator-preparation-v3'","HERE.parent/'native_evaluator'")
 if name=='root_controller.py':
  s=s.replace("MODEL=Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-quant-candidates-c/model-Q8_0.gguf')","MODEL=Path(identity.PROFILE['selection']['q8_path'])")
  s=s.replace("        if result['status']!='complete' or result['denominators']['attempted_cases']!=75:","        if result['status']!='complete' or result['denominators']['attempted_cases']!=75 or result['denominators']['edit_cases']!=43 or result['denominators']['strict_noop_cases']!=32:")
 if name=='guarded_dev_client.py':s=s.replace("Path('/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0')","Path(dev.evaluator.PROFILE['selection']['tokenizer_dir'])")
 if name=='build_closure.py':
  s=s.replace("OLD=HERE.parent/'final-source-closure-preparation-v3'","OLD=HERE.parent.parent/'final-source-closure-preparation-v3'")
  start=s.index("    prior_text=(HERE.parent/'final-native-rehearsal-preparation-v2/client-origins.json').read_text()")
  end=s.index("    prior=json.loads(prior_text)",start)
  s=s[:start]+"    prior_text=(HERE.parent.parent/'final-native-rehearsal-preparation-v3/client-origins.json').read_text()\n    prior_text=prior_text.replace(str(HERE.parent.parent/'final-native-evaluator-preparation-v3'),str(CANDIDATE)).replace(str(HERE.parent.parent/'final-native-rehearsal-preparation-v3'),str(HERE))\n"+s[end:]
 p.write_text(s)
p=H/'native_controller/observe_client.py';s=p.read_text().replace("'python','-I','-S',", "'python','-I','-S','-B',") # actual binary uses full path below
s=s.replace("'/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-I','-S',", "'/home/m0hawk/Documents/Sepalith/.venv-sft/bin/python','-I','-S','-B',");p.write_text(s)
p=H/'native_evaluator/native_identity.py';s=p.read_text().replace("Q8='22401b9f6203cfcb8daa0f5a2919ba74e5e862889050cfb3059ceaf923fb4559'", "from selection_binding import validate_profile\nvalidate_profile(PROFILE)\nQ8=PROFILE['model']['sha256']");p.write_text(s)
p=H/'native_evaluator/rehearse_dev.py';s=p.read_text().replace("'/home/m0hawk/.local/state/sepalith/campaign-20260915/models/SFT-primary-step1000-theta0',", "evaluator.PROFILE['selection']['tokenizer_dir'],")
s=s.replace("if len(rows)!=75 or len({r['id'] for r in rows})!=75 or any(r['split']!='dev' for r in rows):", "if len(rows)!=75 or len({r['id'] for r in rows})!=75 or any(r['split']!='dev' for r in rows) or sum(r['operation']=='no_op' for r in rows)!=32:");p.write_text(s)
# Start from the accepted FP32-accumulation helper, not failed-v1 arithmetic.
p=H/'merge_task_sft_cpu.py';s=(W/'lead/merge_sft_cpu.py').read_text();assert hashlib.sha256(s.encode()).hexdigest()=='e11ca338eaa5e080f81d9b75193f9d73d4cf3c5573311a1830de0259cbe70c3f'
s=s.replace('import os\n','import os\nimport shutil\n').replace("    base = Path(recipe['model_path'])", "    from selection_contract import check_task_recipe\n    check_task_recipe(recipe)\n    assert checkpoint['step'] in (250, 500, 1000)\n    campaign_state = json.loads((args.checkpoint / 'campaign-state.json').read_text())\n    assert campaign_state['identity'] == recipe['identity'] and campaign_state['step'] == checkpoint['step']\n    assert campaign_state['sampler']['consumed_draws'] == checkpoint['step'] * 16\n    assert campaign_state['sampler']['schedule_sha256'] == recipe['draw_schedule']['sha256']\n    base = Path(recipe['model_path'])\n    records = {Path(x['path']).name:x for x in recipe['inputs'] if Path(x['path']).parent == base}\n    assert {'model.safetensors','config.json','generation_config.json','tokenizer.json','tokenizer_config.json'} <= set(records)\n    assert records['model.safetensors']['sha256'] == recipe['identity']['parent']['weights_sha256']")
s=s.replace("    tokenizer.save_pretrained(str(temporary))", "    tokenizer.save_pretrained(str(temporary))\n    for name in ('tokenizer.json','tokenizer_config.json'):\n        shutil.copyfile(base / name, temporary / name)\n        assert digest(temporary / name) == digest(base / name)")
s=s.replace("'sepalith.merged-sft.parent-manifest.v1'", "'sepalith.r2-task-sft.parent-manifest.v1'").replace("'kind': 'merged_sft'", "'kind': 'merged_task_sft'")
s=s.replace("'recipe_sha256': digest(args.recipe),", "'recipe_sha256': digest(args.recipe),\n        'recipe_path': str(args.recipe),\n        'checkpoint_manifest_sha256': digest(args.checkpoint / 'campaign-manifest.json'),\n        'source_cursor': campaign_state['sampler']['consumed_draws'],\n        'parent_model_path': str(base),\n        'tokenizer_original_bytes_restored': True,")
p.write_text(s)
