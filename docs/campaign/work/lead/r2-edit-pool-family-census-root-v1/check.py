import pathlib,json,collections,hashlib,datetime
E=pathlib.Path('/mnt/e/sepalith/campaign-20260915/data-work');P=pathlib.Path(__file__).resolve().parent
FILES=[('original15006','DAT10-finish-source-repair-v3/train-token-rows.jsonl',15006),('candidate616','Semantic763-dedup-integration-v1/final-02/candidate-tokenrows.jsonl',616),('candidate133','Semantic-provider-integration-v1/final-01/candidate-tokenrows.jsonl',133),('candidate4435','Semantic4551-provider-materialization-v1/final-01/candidate-tokenrows.jsonl',4435),('candidate1','Semantic4551-hold-recovery-v1/final-01/candidate-tokenrows.jsonl',1)]
results=[];allfamilies=collections.Counter();allops=collections.Counter()
for name,file,expected in FILES:
 families=collections.Counter();ops=collections.Counter();lengths=collections.Counter();n=0;h=hashlib.sha256()
 for line in (E/file).open('rb'):
  h.update(line);r=json.loads(line);n+=1;families[r.get('family','MISSING')]+=1;ops[r.get('operation','MISSING')]+=1;lengths[next((str(c) for c in (2048,4096,8192,16384,32768,65536,131072) if len(r['input_ids'])<=c),'>128K')]+=1
 assert n==expected;allfamilies.update(families);allops.update(ops);results.append({'cohort':name,'path':str(E/file),'sha256':h.hexdigest(),'rows':n,'families':dict(families),'operations':dict(ops),'smallest_fitting_context':dict(lengths)})
result={'at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'task':'DAT-10','total_review_pool_rows':sum(r['rows'] for r in results),'families':dict(allfamilies),'operations':dict(allops),'cohorts':results,'new_candidates_training_admitted':False,'interpretation':'Census only; counts do not establish label correctness, model quality, or equal value across families.'};(P/'result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
