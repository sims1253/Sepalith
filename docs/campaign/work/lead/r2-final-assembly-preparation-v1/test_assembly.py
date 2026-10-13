import copy, hashlib, json, sys, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
sys.dont_write_bytecode = True
sys.path.insert(0, '/home/m0hawk/Documents/Sepalith/.venv/lib/python3.10/site-packages')
import assemble_inputs as a
class Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.runtime = a.load('assembly_test_runtime', a.ENTRY/'constructor_runtime.py')
        cls.integration, cls.fixtures = cls.runtime.preload()
        cls.modules = {k:sys.modules[n] for k,n in [('builder','dat08_integration_builder'),('families','dat08_integration_raw_families_v3'),('finish','dat08_integration_finish_v4')]}
    def request(self, family, raw):
        return dict(family=family, package_id='synthetic-package', group_id='synthetic-group', source_sha256=hashlib.sha256(raw).hexdigest())
    def test_real_six_builders(self):
        self.assertEqual({r['family'] for r in self.fixtures}, set(a.FAMILIES)|{'format_propagation'})
    def test_real_absence_all_five(self):
        raw=b'x <- 1\n'
        for family in a.FAMILIES:
            with self.subTest(family=family):
                self.assertIsNone(a.probe(self.request(family,raw),{'path':'R/synthetic.R','seed':3407},raw,self.modules))
    def test_real_source_hash_failure_not_skipped(self):
        raw=b'x <- 1\n'; r=self.request('no_op',raw); r['source_sha256']='0'*64
        with self.assertRaises(ValueError): a.probe(r,{'path':'R/synthetic.R','seed':3407},raw,self.modules)
    def test_real_parser_failure_not_skipped(self):
        raw=b'f <- function( {\n'
        with self.assertRaises(ValueError): a.probe(self.request('pipe_rewrite',raw),{'path':'R/synthetic.R','seed':3407},raw,self.modules)
    def test_unexpected_error_not_skipped(self):
        def fail(*args,**kw): raise ValueError('canonical_extractor_failed')
        with self.assertRaisesRegex(ValueError,'canonical_extractor_failed'):
            a.probe(self.request('pipe_rewrite',b'x'),{'path':'R/a.R','seed':3407},b'x',{'builder':SimpleNamespace(build_raw_source_case=fail)})
    def test_selection_determinism_caps_diversity(self):
        selection={'planned_family_ceilings':dict.fromkeys(a.FAMILIES,3),'max_cases_per_group':2,'core_total_ceiling':7}
        rows=[dict(row_id=f'{g}-{f}-{i}',group_id=g,family=f) for g in ['a','b','c','d'] for f in a.FAMILIES for i in range(3)]
        chosen,counts,groups=a.select(rows,selection)
        self.assertEqual(chosen,a.select(list(reversed(rows)),selection)[0]); self.assertEqual(len(chosen),7)
        self.assertEqual(len(groups),4); self.assertLessEqual(max(groups.values()),2); self.assertLessEqual(max(counts.values()),3)
    def test_metadata_final_group_and_alias_validation(self):
        sel={'split_id':'x','global_split_sha256':a.REGISTRY,'requires_weight_and_harness_freeze_receipt':True,
             'planned_family_ceilings':dict.fromkeys(set(a.FAMILIES)|set(a.GAPS),1),
             'source_files':[{'path':'/synthetic/pkg/1/pkg/R/a.R','sha256':'a'*64}],
             'parents':[{'group_id':'f','split':'final_candidate_group','identity':'pkg:pkg','package':'pkg','version':'/synthetic/pkg/1','files':['/synthetic/pkg/1/pkg/R/a.R']}]}
        reg={'split_id':'x','groups':[{'group_id':g,'split':split,'flags':[],'identity_forms':['pkg:'+p]} for g,split,p in [('f','final_candidate_group','pkg'),('t','train_group','training'),('d','dev_group','development')]]}
        sel['parents'][0]['package']='Pkg'
        req,spec,ident=a.metadata_plan(sel,reg); self.assertEqual(len(req),5);self.assertEqual(ident['train']['package_ids'],['training'])
        bad=copy.deepcopy(reg);bad['groups'][0]['split']='train_group'
        with self.assertRaises(ValueError):a.metadata_plan(sel,bad)
        bad=copy.deepcopy(reg);bad['groups'][0]['identity_forms']=['pkg:other']
        with self.assertRaises(ValueError):a.metadata_plan(sel,bad)
    def test_freeze_denied_before_any_metadata_read(self):
        sys.path.insert(0,str(a.ENTRY));import entry_gate
        with patch.object(entry_gate,'release',side_effect=ValueError('freeze denied')),patch.object(a,'checked_json') as read:
            with self.assertRaisesRegex(ValueError,'freeze denied'):
                a.execute(freeze_path='/synthetic/freeze',harness_sha256='h',source_graph_path='/synthetic/graph',source_graph_sha256='g',selection_path='/unread/selection',registry_path='/unread/registry',output_dir='/uncreated/output')
            read.assert_not_called()
if __name__=='__main__':unittest.main(verbosity=2)
