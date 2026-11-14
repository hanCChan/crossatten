"""Metadata admission and frozen W4 algorithm comparison, before any IQ decoding."""
import ast
from common import *

def tree(path):return ast.parse(Path(path).read_text(encoding='utf-8'))
def functions(path):return {n.name:ast.dump(n,include_attributes=False) for n in tree(path).body if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
def main():
    old=ROOT/'research/wisig_w4'
    checks={}
    checks['p_models_byte_equal']=sha(HERE/'p_models.py')==sha(old/'p_models.py')
    checks['fit_aux_AST_equal']=functions(HERE/'p_fit.py')['fit_aux']==functions(old/'p_fit.py')['fit_aux']
    a=functions(HERE/'p_data.py');b=functions(old/'p_data.py')
    checks['data_functions_equal_except_V1_audit_label']=all(a[k]==b[k].replace('mmap_C_V0_admitted_IQ_rows','mmap_C_V1_admitted_IQ_rows') for k in b)
    assert all(checks.values()),checks
    qa=read(HERE/'INDEX_QA.json');qual=read(HERE/'QUALIFICATION.json');plan=read(HERE/'DATA_PLAN.json')
    assert plan['view']['id']=='V1' and len(plan['aux_train_TX'])==72 and len(plan['aux_RX'])==6 and len(plan['cells'])==864
    assert not set(plan['aux_RX'])&set(plan['view']['source_selection_rx'])
    assert plan['aux_train_TX']==read(old/'DATA_PLAN.json')['aux_train_TX']
    write('RECIPE_LOCK.json',dict(status='PASS_W4_A1_ALGORITHM_UNCHANGED',checks=checks,W4_reference_commit='cd597a5',
        new_algorithms=0,lambda_value=0.,V0_pretrained_weights_reused=False,
        copied_source_sha256={n:sha(old/n) for n in ('p_models.py','p_fit.py','p_data.py')},
        IQ_decoded_before_admission=0))
    files=list(HERE.glob('*.py'))+[HERE/n for n in ('PREREGISTRATION.zh-CN.md','DATA_PLAN.json','INDEX_QA.json','QUALIFICATION.json','RECIPE_LOCK.json')]
    historical=read(old/'EXECUTION_REGISTRATION.json')['files_sha256']
    files.extend(ROOT/rel for rel in historical if rel.endswith('.py'))
    write('ADMISSION_REGISTRATION.json',dict(status='METADATA_AND_UNCHANGED_ALGORITHM_REGISTERED_BEFORE_IQ',
        time=now(),files_sha256={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(set(files))},
        permitted='72 fixed auxiliary TX x six V1 fit RX x D1/D2 raw rows0:32; V1 fit_train/checkpoint_val',
        forbidden='V1 held-source IQ before model seal; all other TX/rows/RX/dates; no W5-B'))
    print('LOCK PASS; commit before synthetic or IQ admission')
if __name__=='__main__':main()
