import hashlib, json, tempfile, unittest
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import audit_expanded_union as A

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def geom(path,cursor,window='a'*64):
 g={'schema':A.GEOMETRY_SCHEMA,'source_path':path,'source_sha256':'b'*64,
    'preedit_sha256':'c'*64,'cursor':{'line':cursor,'character':0},
    'replacement_range':{'start':{'line':cursor,'character':0},'end':{'line':cursor,'character':0}},
    'window_sha256':window}
 return g,A.digest_text(A.canonical(g))
def write(p,rows):
 p.write_text(''.join(A.canonical(x)+'\n' for x in rows));return {'path':str(p),'sha256':sha(p),'rows':len(rows)}
def row(i,prompt,target='NO_EDIT'):
 return {'id':i,'prompt_text':prompt,'target_text':target,'split':'train','target_operation':'no_op'}
def prov(i,path='R/x.R',cursor=1,complete=True):
 x={'row_id':i}
 if complete:
  g,d=geom(path,cursor);x.update(source_cursor_geometry=g,source_cursor_geometry_sha256=d)
 return x

class TestAudit(unittest.TestCase):
 def spec(self, cohorts):
  return {'schema':A.SCHEMA,'cohorts':cohorts}
 def cohort(self,d,name,rows,provs,expected=None):
  m=d/(name+'.manifest.json');m.write_text('{}\n')
  t=write(d/(name+'.rows.jsonl'),rows);p=write(d/(name+'.prov.jsonl'),provs)
  return {'name':name,'expected_rows':len(rows) if expected is None else expected,
          'manifest':{'path':str(m),'sha256':sha(m),'rows':0},'token_rows':[t],'provenance_rows':[p]}
 def patch_expected(self, mapping):
  old=A.EXPECTED_COHORTS;A.EXPECTED_COHORTS={k:len(v[0]) for k,v in mapping.items()};self.addCleanup(setattr,A,'EXPECTED_COHORTS',old)
 def audit(self,mapping):
  self.patch_expected(mapping)
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);cs=[self.cohort(d,n,*mapping[n]) for n in mapping]
   return A.audit(self.spec(cs))
 def test_missing_provenance_geometry_is_named_not_dropped(self):
  rows,report=self.audit({'old':([row('a','p')],[prov('a',complete=False)])})
  self.assertEqual(rows[0]['status'],'retained_for_prompt_audit');self.assertIn('missing_source_cursor_geometry',rows[0]['reasons']);self.assertFalse(report['admission_ready'])
 def test_geometry_ambiguity_is_named(self):
  p=prov('a');p['source_cursor_geometry']['cursor']['line']=9
  rows,_=self.audit({'old':([row('a','p')],[p])})
  self.assertTrue(rows[0]['reasons'][0].startswith('ambiguous_source_cursor_geometry:geometry_digest_mismatch'))
 def test_cross_cohort_prompt_contradiction_holds_both(self):
  rows,_=self.audit({'a':([row('a','p','x')],[prov('a')]),'b':([row('b','p','y')],[prov('b',cursor=2)])})
  self.assertEqual({x['status'] for x in rows},{'hold'})
 def test_cross_cohort_geometry_conflict_holds_both(self):
  rows,_=self.audit({'a':([row('a','p1','x')],[prov('a')]),'b':([row('b','p2','y')],[prov('b')])})
  self.assertEqual({x['status'] for x in rows},{'hold'})
 def test_same_file_different_cursor_and_noedit_targets_retained(self):
  rows,report=self.audit({'a':([row('a','p1'),row('b','p2')],[prov('a',cursor=1),prov('b',cursor=2)])})
  self.assertEqual([x['status'] for x in rows],['retained_for_prompt_audit']*2);self.assertTrue(report['admission_ready'])
 def test_same_geometry_same_target_different_prompts_retained(self):
  rows,_=self.audit({'a':([row('a','p1'),row('b','p2')],[prov('a'),prov('b')])})
  self.assertEqual([x['status'] for x in rows],['retained_for_prompt_audit']*2)
 def test_exact_prompt_target_duplicate_only_later_duplicate(self):
  rows,_=self.audit({'a':([row('a','same'),row('b','same')],[prov('a',cursor=1),prov('b',cursor=2)])})
  self.assertEqual([x['status'] for x in rows],['retained_for_prompt_audit','exact_duplicate'])
 def test_unbound_future_cohort_fails_closed_without_dropping_bound(self):
  old=A.EXPECTED_COHORTS;A.EXPECTED_COHORTS={'bound':1,'future':2};self.addCleanup(setattr,A,'EXPECTED_COHORTS',old)
  with tempfile.TemporaryDirectory() as td:
   d=Path(td);c=self.cohort(d,'bound',[row('a','p')],[prov('a')])
   rows,report=A.audit(self.spec([c,{'name':'future','expected_rows':2,'manifest':None,'token_rows':None,'provenance_rows':None}]))
  self.assertEqual(len(rows),1);self.assertEqual(report['status'],'partial_fail_closed');self.assertEqual(report['unbound_cohorts'][0]['rows'],2)
 def test_row_order_join_is_exact(self):
  with self.assertRaisesRegex(A.AuditError,'provenance_row_join'):
   self.audit({'a':([row('a','p')],[prov('wrong')])})

if __name__=='__main__':unittest.main()
