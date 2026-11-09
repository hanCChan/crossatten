"""Public official weights, pinned immutable revision, no credential/tokenizer use."""
import json,hashlib,os
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
from huggingface_hub import HfApi,snapshot_download
HERE=Path(__file__).resolve().parent
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(16<<20),b''):h.update(b)
    return h.hexdigest()
def main():
    manifest=HERE/'MODEL_MANIFEST.json'
    if manifest.exists():
        data=json.loads(manifest.read_text(encoding='utf-8'))
        for file in data['files']:
            if sha(Path(file['path']))!=file['sha256']:raise RuntimeError('Downloaded model hash changed')
        print('Pinned model already verified',data['revision']);return
    info=HfApi(token=False).model_info('Qwen/Qwen3-0.6B',files_metadata=True)
    rev=info.sha
    local=Path(snapshot_download('Qwen/Qwen3-0.6B',revision=rev,token=False,cache_dir=str(HERE/'assets/hf'),
        allow_patterns=['config.json','model*.safetensors','*.safetensors.index.json','LICENSE','README.md'],max_workers=2))
    cfg=json.loads((local/'config.json').read_text(encoding='utf-8'))
    assert cfg['model_type']=='qwen3' and cfg['hidden_size']==1024 and cfg['num_hidden_layers']==28
    files=[dict(name=p.name,path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(local.iterdir()) if p.is_file()]
    data=dict(time=datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds'),repo='Qwen/Qwen3-0.6B',revision=rev,
        local_path=str(local),config=cfg,files=files,tokenizer_downloaded=False,text_generation=False)
    manifest.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n',encoding='utf-8',newline='\n')
    print('Pinned official model',rev,'bytes',sum(v['bytes'] for v in files),flush=True)
if __name__=='__main__':main()
