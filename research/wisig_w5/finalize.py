"""Publish final metadata and check that staged git blobs preserve frozen bytes."""
import argparse
from common import *
def status():
    r=read(HERE/'P_RESULT.json');q=read(HERE/'INDEPENDENT_QA.json');p=read(HERE/'REVIEW_PACKAGE.json');c=r['comparisons']['A1_minus_A0']
    item=dict(status=r['status'],date_Asia_Shanghai='2026-11-14',source_view='V1',branch=git('branch','--show-current'),
        baseline_commit='cd597a5',new_neural_fits=6,reused_A0=3,new_gradient_updates=28200,selected_weight_files=9,
        source_selection_packets_per_model=11500,summary=r['summary'],comparison=c,gates=r['gates'],
        independent_checks=q['checks_passed'],portable_checks=p['portable_checks'],review_archive=p,
        W5_B_started=False,A2_started=False,actual_target_CRD_D3_D4_historical_held6_IQ_reads=0,LoRa_modified=False)
    dest=ROOT/'research/WISIG_W5_A_STATUS_2026_11_14.json'
    dest.write_text(json.dumps(item,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    header=f"""<!-- W5_A_2026_11_14 -->
2026-11-14 最新：[W5-A固定A1跨视图复现](research/wisig_w5/README.zh-CN.md)。V1源侧A0/A1的RX等权Accuracy为{r['summary']['A0']['RX_equal_accuracy_pct']:.4f}%/{r['summary']['A1']['RX_equal_accuracy_pct']:.4f}%，配对差{c['mean_gain_pp']:+.4f}pp，{c['positive_seeds']}/3 seed正向；最差RX的seed均值差{c['worst_RX_seed_mean_gain_pp']:+.4f}pp。正式决策：{r['status']}。6次新拟合/28,200更新，复用3份A0；[机器状态](research/WISIG_W5_A_STATUS_2026_11_14.json)、[含9份所选权重的审阅包](output/review/wisig_w5_review.zip)。这是已开发内部源RX的跨视图复现，非历史新盲测或目标CRD。W5-B未启动，A2与旧STOP、LoRa均保持原结论。

"""
    for name in ('README.md','PLAN.md','RESULTS.md'):
        path=ROOT/name;text=path.read_text(encoding='utf-8')
        if '<!-- W5_A_2026_11_14 -->' not in text:path.write_text(header+text,encoding='utf-8',newline='\n')
    print('Wrote final W5 status and top-level result links')
def checkout():
    files=set()
    for name in ('ADMISSION_REGISTRATION.json','EXECUTION_REGISTRATION.json','P_EVALUATION_REGISTRATION.json'):
        files.update(read(HERE/name)['files_sha256'])
        files.add((HERE/name).relative_to(ROOT).as_posix())
    files.update(i['relative'] for i in read(HERE/'REPLAY_FILES.json')['files'])
    files.update(p.relative_to(ROOT).as_posix() for p in HERE.iterdir() if p.is_file() and p.name!='GIT_CHECKOUT_QA.json')
    for rel in sorted(files):
        blob=subprocess.check_output(['git','-C',str(ROOT),'cat-file','blob',':'+rel])
        assert hashlib.sha256(blob).hexdigest()==sha(ROOT/rel),'Staged byte mismatch: '+rel
    names=git('diff','--name-only','cd597a5','--').splitlines()
    allowed={'.gitattributes','README.md','PLAN.md','RESULTS.md','research/WISIG_W5_A_STATUS_2026_11_14.json','output/review/wisig_w5_review.zip'}
    assert all(n in allowed or n.startswith('research/wisig_w5/') for n in names),names
    assert not git('diff','--name-only','cd597a5','--','publication/lora_hstem')
    write('GIT_CHECKOUT_QA.json',dict(status='PASS',frozen_and_replay_blobs_checked=len(files),
        staged_blobs_byte_identical=True,old_stages_or_LoRa_changed=False,diff_base='cd597a5',files=sorted(files)))
    print('PASS',len(files),'staged exact bytes; changes confined to W5 and result navigation')
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('command',choices=('status','checkout'));globals()[p.parse_args().command]()
