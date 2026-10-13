#!/usr/bin/env python3
import importlib.util,json,tempfile
from pathlib import Path
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('all_eligible',HERE/'materialize_all_eligible.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class FakeEncoding:
    def __init__(self,ids): self.ids=ids
class FakeTokenizer:
    def encode(self,text,add_special_tokens=False): return FakeEncoding([2+ord(x) for x in text])
    def decode(self,ids,skip_special_tokens=False): return ''.join(chr(x-2) for x in ids)

class FakeRawCpt:
    @staticmethod
    def allowed_license(value): return value=='MIT'
    @staticmethod
    def chunks(ids,size):
        start=0
        while True:
            carry=ids[start-1:start] if start else [];capacity=size-2-len(carry);end=min(len(ids),start+capacity);terminal=end==len(ids)
            seq=[0]+carry+ids[start:end]+[1];labels=[-100]*(1+len(carry))+ids[start:end]+([1] if terminal else [-100])
            yield {'input_ids':seq,'labels':labels,'attention_mask':[1]*len(seq),'source_token_start':start,'source_token_end':end,'token_start':start,'token_end':end,'document_token_count':len(ids),'overlap_context_tokens':len(carry),'is_document_end':terminal,'supervised_tokens':sum(x!=-100 for x in labels)}
            if terminal:return
            start=end

def package(root,name='pkg',payload=b'x<-1\n'):
    base=root/name/'1.0'/name;(base/'R').mkdir(parents=True)
    (base/'DESCRIPTION').write_text(f'Package: {name}\nLicense: MIT\n')
    (base/'R/code.R').write_bytes(payload)
    return {'name':name,'group_id':'g-train','split':'train_group'}

class MaterializerTest(unittest.TestCase):
    def test_inventory_has_no_old_four_mib_file_cap(self):
        with tempfile.TemporaryDirectory() as td:
            old=m.NORMALIZED;m.NORMALIZED=Path(td)
            try:
                entry=package(Path(td),payload=b'x'*(4*1024*1024+1))
                rows,repairs,categories=m.inventory_package(entry,{'pkg':('g-train','train_group')},{'g-train':'cpt_train'})
                self.assertEqual(len(rows),1);self.assertGreater(rows[0]['bytes'],4*1024*1024)
                self.assertFalse(repairs);self.assertEqual(categories['regular_R'],1)
            finally:m.NORMALIZED=old

    def test_heldout_group_is_rejected_before_payload_inventory(self):
        with tempfile.TemporaryDirectory() as td:
            old=m.NORMALIZED;m.NORMALIZED=Path(td)
            try:
                entry=package(Path(td))
                with self.assertRaisesRegex(ValueError,'heldout'):
                    m.inventory_package(entry,{'pkg':('g-train','train_group')},{'g-train':'cpt_validation'})
            finally:m.NORMALIZED=old

    def test_empty_regular_R_is_a_named_degenerate_exclusion(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);old=m.NORMALIZED;m.NORMALIZED=root/'normalized';entry=package(m.NORMALIZED,payload=b'')
            output=root/'out';(output/'groups').mkdir(parents=True);(output/'.staging').mkdir()
            try:
                receipt=m.process_group(output,775,entry,FakeTokenizer(),FakeRawCpt(),{'pkg':('g-train','train_group')},{'g-train':'cpt_train'},set(),set())
                self.assertEqual(receipt['status'],'complete');self.assertEqual(receipt['counts']['excluded_empty_R_file_degenerate'],1)
                self.assertEqual(receipt['counts'].get('repair_items',0),0)
            finally:m.NORMALIZED=old

    def test_atomic_group_receipt_reconciles_without_duplicate(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);old=m.NORMALIZED;m.NORMALIZED=root/'normalized';entry=package(m.NORMALIZED)
            output=root/'out';(output/'groups').mkdir(parents=True);(output/'.staging').mkdir()
            try:
                receipt=m.process_group(output,775,entry,FakeTokenizer(),FakeRawCpt(),{'pkg':('g-train','train_group')},{'g-train':'cpt_train'},set(),set())
                self.assertEqual(receipt['status'],'complete');self.assertEqual(receipt['counts']['documents'],1)
                seen=set();receipts,totals=m.committed(output,seen)
                self.assertEqual(set(receipts),{775});self.assertEqual(totals['documents'],1);self.assertEqual(len(seen),1)
                with self.assertRaisesRegex(ValueError,'already exists'):
                    m.process_group(output,775,entry,FakeTokenizer(),FakeRawCpt(),{'pkg':('g-train','train_group')},{'g-train':'cpt_train'},seen,set())
            finally:m.NORMALIZED=old

    def test_frozen_order_preflight_names_exact_remaining_extent(self):
        entries,_,_,source=m.preflight(m.DEFAULT_OUTPUT)
        self.assertEqual(len(entries),8092);self.assertEqual(source['caps'],{'wall_time':None,'group_tokens':None,'package_tokens':None,'file_bytes':None})

    def test_narrow_source_migration_rejects_other_contract_changes(self):
        _,_,_,current=m.preflight(m.DEFAULT_OUTPUT);prior=dict(current)
        prior['materializer_sha256']=json.loads(m.SOURCE_MIGRATION.read_text())['from_materializer_sha256']
        with tempfile.TemporaryDirectory() as td:
            output=Path(td);(output/'groups').mkdir()
            m.admit_source_migration(prior,current,output)
            self.assertEqual(json.loads((output/'run-manifest.json').read_text()),current)
            changed=dict(prior);changed['groups']=8091
            with self.assertRaisesRegex(ValueError,'changed more'):
                m.admit_source_migration(changed,current,output)

if __name__=='__main__':unittest.main()
