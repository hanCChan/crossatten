"""Single-GPU sequential controller. Wait for S15; preserve scientific failures."""
import os,sys,time,subprocess,traceback
from s16_core import HERE,ROOT,S15,read,write,now,validate,commit,lock,process_alive

def status(state,**fields):write(HERE/'STATUS.json',dict(time=now(),status=state,**fields))

def child(script,*args,log):
    if (HERE/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED at job boundary')
    env=dict(os.environ,PYTHONUTF8='1',CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUBLAS_WORKSPACE_CONFIG=':4096:8')
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-X','utf8','-u',str(HERE/script),*args],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)

def wait_s15():
    while True:
        if (HERE/'STOP_REQUESTED').exists():raise RuntimeError('STOP_REQUESTED while waiting')
        path=S15/'STATUS.json';state=read(path) if path.exists() else {}
        pid_path=S15/'execution.pid';alive=process_alive(int(pid_path.read_text().strip())) if pid_path.exists() else False
        if state.get('status')=='COMPLETE' and not alive:
            from s16_core import committed
            committed(path,root=S15.parent)
            return
        if state.get('status','').startswith('STOP') or (not alive and state.get('status')!='COMPLETE'):
            raise RuntimeError('S15 failed/interrupted; resolve its recorded failure before S16 GPU use')
        status('WAITING_FOR_S15',s15_status=state,s15_process_alive=alive)
        time.sleep(30)

def run_matrix(phase,front=False):
    from s16_train import run_id
    matrix=f'jobs_{phase}.json';jobs=read(HERE/matrix)
    for i,j in enumerate(jobs):
        out=HERE/'runs'/run_id(j)
        if (out/'metrics.json').exists():continue
        status('TRAINING_RF_FRONT' if front else 'TRAINING_SOURCE_TAIL',phase=phase,job=j,
            completed=sum((HERE/'runs'/run_id(k)/'metrics.json').exists() for k in jobs),registered=len(jobs))
        print(now(),'FIT',run_id(j),flush=True)
        child('s16_train.py','--matrix',matrix,'--index',str(i),*(['--front'] if front else []),log=out/'stdout.log')

def phase_report(phase):
    child('s16_reports.py',phase,log=HERE/'runs'/f'report_{phase}.log')
    evidence=[HERE/f'decision_{phase}.json',HERE/f'{phase.upper()}_RESULTS.zh-CN.md']+list(HERE.glob(f'{phase}_*.csv'))
    commit(evidence,f'S16 freeze {phase} complete paired source-only outcomes and gate')
    return read(HERE/f'decision_{phase}.json')

def finalize(last):
    decisions={p.stem:read(p) for p in sorted(HERE.glob('decision_*.json'))}
    result=dict(time=now(),status='READY_TO_PREREGISTER_FINAL_CRD' if last['selected'] and last['phase']=='confirm2' else 'STOP_SOURCE_GATE',
        decisions=decisions,real_target_access=False,fit_count=len(list((HERE/'runs').glob('*/metrics.json'))),
        final_target_evaluation_automatic=False,reason='Source gates only; future target matrix requires separate freeze after any passing confirmation')
    write(HERE/'summary.json',result)
    lines=['# S16 最终源域结论','',f"状态：{result['status']}；完成{result['fit_count']}个训练。",'',
        '各阶段配对均值、最差RX、Macro-F1、训练差距和成本见下列结果文件。源域留出RX不等于真实目标CRD。']
    for phase in ['pilot','confirm1','confirm2']:
        if (HERE/f'decision_{phase}.json').exists():lines.append(f'- [{phase}]({phase.upper()}_RESULTS.zh-CN.md)')
    lines+=['','解释范围：小型Qwen3与同深度/近似参数Transformer的差异是结构组合证据；相对小型R0的差异仍混有深度、宽度和规模。',
        '预训练冻结骨干仅由R3−R4判断；若R5进入独立确认，还须R5−R6支持同LoRA配方下的预训练贡献。不能把收益单独归因为语言知识。',
        '任何源域门槛未通过即停止，不通过目标RX或date3–4调参。全部失败、负收益与资源门槛保留。未单独实施深度消融，因此不能唯一回答深度的因果作用。',
        '推理成本见P0_ENGINEERING.json（batch1/16、尾部及完整RF路径、合成IQ）；训练成本与活跃/可训练参数见阶段CSV。']
    (HERE/'RESULTS.zh-CN.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    status('COMPLETE',research_decision=result['status'],fit_count=result['fit_count'])
    commit([HERE/'summary.json',HERE/'RESULTS.zh-CN.md',HERE/'STATUS.json'],'S16 finalize bounded RF-Qwen3 source exploration')

def run():
    validate('design.json');wait_s15()
    status('ENGINEERING_P0');child('engineering.py',log=HERE/'runs/engineering.log')
    commit([HERE/'P0_ENGINEERING.json'],'S16 freeze synthetic resource and gradient checks before RF pilot')
    child('s16_register.py','pilot',log=HERE/'runs/registration.log')
    run_matrix('pilot');decision=phase_report('pilot')
    for phase,front in [('confirm1','front_r1'),('confirm2','front_r2')]:
        if not decision['selected']:break
        child('s16_register.py',phase,log=HERE/'runs/registration.log')
        run_matrix(front,True);run_matrix(phase);decision=phase_report(phase)
    finalize(decision)

if __name__=='__main__':
    with lock():
        try:run()
        except Exception:
            write(HERE/'execution_failure.json',dict(time=now(),traceback=traceback.format_exc(),automatic_retry=False))
            status('STOP_TECHNICAL_REVIEW');raise
