from pathlib import Path
import copy,hashlib,json,sqlite3,struct,sys,tempfile,unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'source/experiments/training'))
from prefix_extension_contract import verify_consumed_prefix

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
class PrefixTest(unittest.TestCase):
 def setUp(self):
  self.t=tempfile.TemporaryDirectory();self.d=Path(self.t.name)
  self.old_ids=[f'old{i:02}' for i in range(20)];self.new_ids=self.old_ids+[f'new{i:02}' for i in range(12)]
  self.old_cache=self.cache('old',self.old_ids,self.old_ids+self.old_ids[:12]);self.new_cache=self.cache('new',self.new_ids,self.new_ids)
  self.old_schedule=self.schedule('old-schedule',self.old_ids,self.old_ids[:12]);self.new_schedule=self.schedule('new-schedule',self.new_ids,[])
  self.source={'transition':{'global_optimizer_step_offset':66},'cohort':{'streaming_cache':{'path':str(self.old_cache),'manifest_sha256':sha(self.old_cache/'manifest.json')},'draw_schedule':{'path':str(self.old_schedule),'sha256':sha(self.old_schedule)}}}
  self.dest={'transition':{'destination_sampler':'preserve_verified_prefix','global_optimizer_step_offset':66,'source_global_step':67,'source_cursor':16},'cohort':{'streaming_cache':{'path':str(self.new_cache),'manifest_sha256':sha(self.new_cache/'manifest.json')},'draw_schedule':{'path':str(self.new_schedule),'sha256':sha(self.new_schedule)}},'runtime':{'mandatory_stop_step':194,'max_steps':200}}
  self.state={'sampler':{'method':'sequential_frozen_draw_schedule_stage_local','global_optimizer_step_offset':66,'global_step':67,'stage_cursor':16,'cursor':16,'draw_schedule_sha256':sha(self.old_schedule),'ignore_data_skip':True}}
 def cache(self,name,ids,draws):
  root=self.d/name;root.mkdir();offset=0;inputs=[];labels=[];source_lines=[]
  db=sqlite3.connect(root/'index.sqlite3');db.executescript('create table rows(ordinal integer primary key,row_id text not null unique,token_offset integer not null,length integer not null,document_id text not null,package text not null,source_sha256 text not null,row_sha256 text not null);create table documents(document_id text primary key,package text not null,source_sha256 text not null unique,chunks integer not null,tokens integer not null,token_stream_sha256 text not null);')
  for i,rid in enumerate(ids):
   tok=[0,100+i,1];lab=[-100,100+i,1];src=hashlib.sha256(rid.encode()).hexdigest();pkg=f'p{i%2}';row={'row_id':rid,'document_id':rid,'package':pkg,'source_sha256':src,'input_ids':tok,'labels':lab,'overlap_context_tokens':0,'supervised_tokens':2};line=(json.dumps(row,separators=(',',':'))+'\n').encode();source_lines.append(line);rh=hashlib.sha256(line).hexdigest()
   db.execute('insert into rows values(?,?,?,?,?,?,?,?)',(i,rid,offset,3,rid,pkg,src,rh));db.execute('insert into documents values(?,?,?,?,?,?)',(rid,pkg,src,1,3,hashlib.sha256(struct.pack('<3i',*tok)).hexdigest()));offset+=3;inputs+=tok;labels+=lab
  db.commit();db.close();source=root/'rows.jsonl';source.write_bytes(b''.join(source_lines));(root/'input_ids.i32le').write_bytes(struct.pack(f'<{len(inputs)}i',*inputs));(root/'labels.i32le').write_bytes(struct.pack(f'<{len(labels)}i',*labels))
  ordinal={x:i for i,x in enumerate(ids)};(root/'draw_ordinals.i64le').write_bytes(struct.pack(f'<{len(draws)}Q',*(ordinal[x] for x in draws)))
  files={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in root.iterdir()}
  m={'schema':'sepalith.sft11.cpt-streaming-cache.v1','status':'complete','max_sequence_tokens':16,'source':{'rows':{'path':str(source),'sha256':sha(source)}},'counts':{'rows':len(ids),'documents':len(ids),'packages':2,'input_tokens':len(inputs),'payload_tokens':len(ids),'loss_tokens':2*len(ids),'draws':len(draws)},'files':files};(root/'manifest.json').write_text(json.dumps(m));return root
 def schedule(self,name,unique,replays):
  p=self.d/(name+'.json');ids=unique+replays;v={'effective_batch':16,'row_ids':ids,'replay_row_ids':replays,'coverage':{'draws':len(ids),'unique_rows':len(unique),'updates':len(ids)//16}};p.write_text(json.dumps(v));return p
 def repin(self):self.dest['cohort']['draw_schedule']['sha256']=sha(self.new_schedule)
 def test_full_prefix_and_old_unconsumed_replays_may_drop(self):
  self.assertEqual(verify_consumed_prefix(self.source,self.dest,self.state),16)
 def test_bad_unique_prefix_rejected(self):
  x=json.loads(self.new_schedule.read_text());x['row_ids'][0],x['row_ids'][1]=x['row_ids'][1],x['row_ids'][0];self.new_schedule.write_text(json.dumps(x));self.repin()
  with self.assertRaisesRegex(ValueError,'old unique'):verify_consumed_prefix(self.source,self.dest,self.state)
 def test_cursor_offset_and_future_stop_rejected(self):
  for mutate,pattern in ((lambda:self.state['sampler'].__setitem__('cursor',0),'cursor'),(lambda:self.dest['transition'].__setitem__('global_optimizer_step_offset',65),'offset'),(lambda:self.dest['runtime'].__setitem__('mandatory_stop_step',67),'future stop')):
   s,d=copy.deepcopy(self.state),copy.deepcopy(self.dest);mutate()
   with self.assertRaisesRegex(ValueError,pattern):verify_consumed_prefix(self.source,self.dest,self.state)
   self.state,self.dest=s,d
 def test_changed_old_token_prefix_rejected_even_when_rehashed(self):
  p=self.new_cache/'input_ids.i32le';raw=bytearray(p.read_bytes());raw[4]^=1;p.write_bytes(raw)
  m=json.loads((self.new_cache/'manifest.json').read_text());m['files']['input_ids.i32le']['sha256']=sha(p);(self.new_cache/'manifest.json').write_text(json.dumps(m));self.dest['cohort']['streaming_cache']['manifest_sha256']=sha(self.new_cache/'manifest.json')
  with self.assertRaisesRegex(ValueError,'cache arrays|old binary prefix'):verify_consumed_prefix(self.source,self.dest,self.state)
 def test_dropped_old_row_rejected(self):
  x=json.loads(self.new_schedule.read_text());x['row_ids'][19]='new00';self.new_schedule.write_text(json.dumps(x));self.repin()
  with self.assertRaises(ValueError):verify_consumed_prefix(self.source,self.dest,self.state)
if __name__=='__main__':unittest.main()
