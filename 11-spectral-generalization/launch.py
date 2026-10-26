"""Bounded launcher. Every attempt is immutable; no automatic fit retries."""
import argparse
import fcntl
import json
import os
import subprocess
import sys
import threading
from core import HERE, ROOT, validate, save
from training import fit,evaluate


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--jobs")
    p.add_argument("--job")
    p.add_argument("--gpus",default="0,1,2,3")
    p.add_argument("--per-gpu",type=int,default=2)
    p.add_argument("--evaluate",action="store_true")
    a=p.parse_args()
    if a.job:
        (evaluate if a.evaluate else fit)(json.loads(a.job))
        return
    jobs=json.loads((HERE/a.jobs).read_text())
    for j in jobs:
        validate(j)
    if len({j["id"] for j in jobs})!=len(jobs):
        raise ValueError("duplicate job")
    status=dict(total=len(jobs),done=[],skipped=[],failed=[])
    status_path=HERE/(a.jobs.replace(".json","") + (".evaluate" if a.evaluate else "") + ".status.json")
    lock,cursor=threading.Lock(),iter(jobs)

    def worker(gpu):
        while True:
            with lock:
                j=next(cursor,None)
            if j is None:
                return
            out=HERE/"runs"/j["phase"]/j["id"]
            out.mkdir(parents=True,exist_ok=True)
            rc=-1
            try:
                with (out/"execution.lock").open("a") as f:
                    fcntl.flock(f,fcntl.LOCK_EX)
                    suffix="target_metrics.json" if a.evaluate else "metrics.json"
                    if (out/suffix).exists():
                        with lock:
                            status["skipped"].append(j["id"])
                            save(status_path,status)
                        continue
                    marker=out/("target_attempt.json" if a.evaluate else "fit_attempt.json")
                    if marker.exists():
                        raise ValueError("prior failed/incomplete attempt; retry not authorized")
                    save(marker,dict(job=j,gpu=gpu))
                    cmd=[sys.executable,__file__,"--job",json.dumps(j)]
                    if a.evaluate:
                        cmd.append("--evaluate")
                    env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS="2",
                             MKL_NUM_THREADS="2",PYTHONPATH=str(ROOT/"lib"))
                    with (out/("target_stdout.log" if a.evaluate else "stdout.log")).open("w") as log:
                        rc=subprocess.call(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
            except Exception as exc:
                print(j["id"],str(exc),flush=True)
            with lock:
                status["done" if rc==0 else "failed"].append(j["id"])
                save(status_path,status)
            print("gpu",gpu,"rc",rc,j["id"],flush=True)
    threads=[threading.Thread(target=worker,args=(g,)) for g in a.gpus.split(",") for _ in range(a.per_gpu)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if status["failed"]:
        raise SystemExit("training failed; no advancement")


if __name__=="__main__":
    main()
