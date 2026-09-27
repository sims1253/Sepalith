"""Unadmitted R2 TRAIN reward: exact target AND applied-buffer syntax.

Uses the actual pinned campaign reward's token/protocol checks. No R evaluation,
network, model loading, tree-similarity reward or semantic-equivalence claim.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import inspect
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import Any, Callable, Mapping, Sequence

from campaign_rl_train import CampaignPRM03Reward, RLTrainError, line_f1
from sepalith.campaign_protocol import PromptContext, parse_output, utf16_to_codepoint_column

POLICY_ID = 'r2_applied_buffer_parse_and_exact_v1'
LEGACY_TRAINER_SHA256 = '78d27aa98cebc80292d1871a39821eee5a1a705b5a8f412ce270d1707d724443'
LEGACY_REWARD_EXCERPT_SHA256 = 'c3f55dc6061a700ab8ac788c6bd0ea6c13af2ce97240813360582804ef016101'
PROTOCOL_SHA256 = '5a869329be74d8177769901bc771aa3dc6e19dad1f2d97fca149e5c38f840156'
EXACT_REWARD = 1.2
ADMISSION_POLICY = MappingProxyType({'id':POLICY_ID,'exact_reward':EXACT_REWARD,'other_reward':0.0,'line_f1_reward':False,'require_gold_parse':True,'require_prediction_parse':True,'semantic_alternatives':'unverified_no_credit'})
MAX_DOCUMENT_BYTES = 2 * 1024 * 1024
MAX_TOTAL_DOCUMENT_BYTES = 256 * 1024 * 1024

class BindingError(RLTrainError):
    """Invalid or incomplete TRAIN source evidence; stop before training."""

def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def canonical(value: Any) -> str:
    return json.dumps(value,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False)

def context_key(context: PromptContext) -> str:
    return sha_text(canonical(context.to_dict()))

def offset(document: str, position: Any) -> int:
    breaks=list(re.finditer(r'\r\n|\n|\r',document))
    starts=[0]+[m.end() for m in breaks]
    ends=[m.start() for m in breaks]+[len(document)]
    line,character=position.line,position.character
    if type(line) is not int or type(character) is not int or not 0<=line<len(starts) or character<0:
        raise BindingError('replacement position outside bound document')
    try:
        column=utf16_to_codepoint_column(document[starts[line]:ends[line]],character)
    except ValueError as error:
        raise BindingError('replacement UTF16 column is invalid') from error
    return starts[line]+column

@dataclass(frozen=True)
class DocumentBinding:
    row_id: str
    family: str
    package_id: str
    context_sha256: str
    target_operation: str
    target_body_sha256: str
    document: str
    document_sha256: str
    start: int
    end: int
    newline: str
    expected_lines: tuple[str,...]
    expected_document_sha256: str

    def apply(self, lines: Sequence[str]) -> str:
        result=self.document[:self.start]+self.newline.join(lines)+self.document[self.end:]
        if len(result.encode('utf-8'))>MAX_DOCUMENT_BYTES:
            raise BindingError('applied R buffer exceeds candidate byte ceiling')
        return result


def bind_train_records(records: Sequence[Any], parse_r: Callable[[str],bool]) -> Mapping[str,DocumentBinding]:
    """Accept actual validated RLTrainRecord objects; never reconstruct missing source.

    Root must bind the corrected row/sidecar/source artifacts in recipe identity.
    This function checks that their full pre-edit buffers and exact targets agree.
    Missing source or an unparsable gold buffer is an admission failure, not reward0.
    """
    if not records:
        raise BindingError('no TRAIN records')
    bindings={};total=0
    for record in records:
        row=record.row; context=record.context; identity=record.source_identity
        if row.get('split')!='train':
            raise BindingError('reward bindings accept TRAIN rows only')
        row_id=row.get('id')
        if not isinstance(row_id,str) or not row_id or row_id in bindings:
            raise BindingError('TRAIN row ID is missing or duplicated')
        try:
            selection=identity['source_provenance']['selection_source']
            document=selection['document_text']
        except (KeyError,TypeError) as error:
            raise BindingError('TRAIN row lacks full bound pre-edit document: '+row_id) from error
        if not isinstance(document,str):
            raise BindingError('bound document must be text')
        size=len(document.encode('utf-8'));total+=size
        if size>MAX_DOCUMENT_BYTES or total>MAX_TOTAL_DOCUMENT_BYTES:
            raise BindingError('candidate document byte ceiling exceeded')
        rr=context.replacement_range;digest=sha_text(document)
        if digest!=selection.get('content_sha256') or digest!=rr.content_sha256:
            raise BindingError('bound document hash differs from selection/range')
        capture=record.capture
        if (capture.content_sha256!=digest or capture.uri!=rr.uri or capture.version!=rr.document_version):
            raise BindingError('context capture differs from bound range')
        if selection.get('document_version',rr.document_version)!=rr.document_version:
            raise BindingError('source document version differs from bound range')
        start,end=offset(document,rr.start),offset(document,rr.end)
        if start>end:
            raise BindingError('reversed replacement range')
        selected=document[start:end].replace('\r\n','\n').replace('\r','\n')
        if selected!='\n'.join(context.region_old):
            raise BindingError('selected full-document text differs from region_old')
        operation=row['target_operation'];body=row['target_body_text']
        if operation=='no_op':
            if body!='[NO_EDIT]':raise BindingError('no-op target must be canonical')
            expected=tuple(context.region_old)
        elif operation=='delete':
            if body:raise BindingError('delete target must have empty body')
            expected=()
        elif operation=='replace':expected=tuple(body.split('\n'))
        else:raise BindingError('unknown target operation')
        # Use the same protocol parser, including normalization of copied old text.
        gold=parse_output(body+'\n>>>>>>> UPDATED',context)
        if gold.status!='accepted':raise BindingError('gold target fails protocol parser')
        golden_lines=tuple(context.region_old) if gold.operation=='no_op' else tuple(gold.body)
        if golden_lines!=expected:raise BindingError('gold protocol semantics mismatch')
        newline='\r\n' if context.document_eol=='crlf' else '\n'
        temp=DocumentBinding(row_id,str(row['family']),str(row['package_id']),context_key(context),operation,sha_text(body),document,digest,start,end,newline,expected,'')
        expected_document=temp.apply(expected)
        if parse_r(expected_document) is not True:
            raise BindingError('exact TRAIN target leaves an unparsable R buffer: '+row_id)
        bindings[row_id]=DocumentBinding(**{**temp.__dict__,'expected_document_sha256':sha_text(expected_document)})
    return MappingProxyType(bindings)

def prepare_reward_bindings(records, admitted_policy, parse_r):
    if admitted_policy != dict(ADMISSION_POLICY):
        raise BindingError('root recipe has not bound this exact candidate reward policy')
    return bind_train_records(records,parse_r)

class AppliedDocumentExactReward(CampaignPRM03Reward):
    __name__='campaign_r2_applied_r_exact_reward'

    def __init__(self, *, bindings: Mapping[str,DocumentBinding], parse_r: Callable[[str],bool], **kwargs: Any):
        # Pin unchanged reward implementation, so a reviewed factory-only trainer
        # patch does not invalidate its own whole-file ancestor hash. Root must
        # bind the entire NEW trainer/dependency graph separately in recipe identity.
        excerpt=inspect.getsource(line_f1)+inspect.getsource(CampaignPRM03Reward)
        protocol_source=Path(inspect.getfile(parse_output))
        if sha_text(excerpt)!=LEGACY_REWARD_EXCERPT_SHA256 or hashlib.sha256(protocol_source.read_bytes()).hexdigest()!=PROTOCOL_SHA256:
            raise BindingError('legacy reward implementation/protocol source pin mismatch')
        if not bindings:raise BindingError('no admitted TRAIN document bindings')
        self.bindings=MappingProxyType(dict(bindings));self.parse_r=parse_r
        super().__init__(**kwargs)

    def score_one(self, context_value, target_operation, target_body_text, generated_ids, *, row_id='',family='',package_id=''):
        context=context_value if isinstance(context_value,PromptContext) else PromptContext.from_mapping(context_value)
        binding=self.bindings.get(row_id)
        if binding is None:raise BindingError('reward row is outside admitted TRAIN bindings')
        if (binding.context_sha256!=context_key(context) or binding.target_operation!=target_operation or binding.target_body_sha256!=sha_text(target_body_text) or binding.family!=family or binding.package_id!=package_id):
            raise BindingError('reward row/context/target/provenance differs from binding')
        _,record=super().score_one(context,target_operation,target_body_text,generated_ids,row_id=row_id,family=family,package_id=package_id)
        record.update({'reward_policy':POLICY_ID,'legacy_reward_diagnostic':record['reward'],'legacy_line_f1_diagnostic':record['line_f1'],'line_f1_used_for_reward':False,'applied_r_parse':None,'semantic_status':'invalid_protocol','document_sha256':binding.document_sha256,'applied_document_sha256':None,'reward':0.0})
        if record['protocol_valid']:
            ids=self._ids(generated_ids,0)
            parsed=parse_output(self._decode(ids[:-1]),context)
            if parsed.status!='accepted' or parsed.operation!=record['operation']:
                raise BindingError('repeated decoder/protocol result changed')
            lines=tuple(context.region_old) if parsed.operation=='no_op' else tuple(parsed.body)
            document=binding.apply(lines)
            parse_valid=self.parse_r(document)
            if type(parse_valid) is not bool:raise BindingError('parser must return a boolean')
            exact=lines==binding.expected_lines
            if exact!=record['exact_region']:raise BindingError('base exact-target result differs from binding')
            record.update({'applied_r_parse':parse_valid,'applied_document_sha256':sha_text(document),'semantic_status':'exact_reference_parse_valid' if parse_valid and exact else 'nonexact_parse_valid_semantics_unverified' if parse_valid else 'applied_buffer_parse_invalid','reward':EXACT_REWARD if parse_valid and exact else 0.0})
        return record['reward'],record

    def __call__(self, *, split: Sequence[str], id: Sequence[str], **kwargs):
        if len(split)!=len(kwargs['completion_ids']) or len(id)!=len(split) or any(x!='train' for x in split):
            raise BindingError('reward batch must contain aligned explicit TRAIN splits and IDs')
        return super().__call__(id=id,**kwargs)
