"""RF embeddings -> last CLS; architecture and pretrained-weight effects separated."""
import hashlib
import copy
import torch
from torch import nn
from s16_core import HERE,read
from transformers import Qwen3Config,Qwen3Model
from peft import LoraConfig,get_peft_model,TaskType
from models import PlainMHA
from legacy_rfhstu.models import RFHSTUEncoder
from xa.train import stable_seed

ARMS=['R0_mha','R0_hstu','R1_qwen_small','R2_transformer','R3_pretrained_frozen','R4_random_frozen','R5_pretrained_lora','R6_random_lora']
def tiny_config():
    return Qwen3Config(vocab_size=1,hidden_size=128,intermediate_size=384,num_hidden_layers=4,num_attention_heads=4,
        num_key_value_heads=2,head_dim=32,max_position_embeddings=64,pad_token_id=None,bos_token_id=None,eos_token_id=None,
        rope_theta=1000000.,rms_norm_eps=1e-6,attention_dropout=0.,use_cache=False,tie_word_embeddings=False,
        sliding_window=None,use_sliding_window=False,max_window_layers=4)

def drop_vocabulary(model):
    before=sum(p.numel() for p in model.parameters());embed=model.get_input_embeddings()
    discarded=0 if embed is None else sum(p.numel() for p in embed.parameters())
    model.set_input_embeddings(None);model.config.use_cache=False
    return dict(instantiated_params_before_pruning=before,discarded_embedding_params=discarded,lm_head_instantiated=False,
        backbone_params_after_pruning=sum(p.numel() for p in model.parameters()))

def tensor_hash(module):
    h=hashlib.sha256()
    for name,p in module.state_dict().items():
        h.update(name.encode());h.update(str(p.dtype).encode());h.update(p.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()

class ClassicTail(nn.Module):
    def __init__(self,kind,classes):
        super().__init__();self.encoder=RFHSTUEncoder(64,num_patches=32,dim=64,depth=2)
        self.pre_ln=nn.LayerNorm(64);self.head=nn.Linear(64,classes)
        if kind=='R0_mha':
            with torch.random.fork_rng(devices=[]):self.encoder.blocks=nn.ModuleList([PlainMHA(64,4) for _ in range(2)])
    def forward(self,x):
        h=self.pre_ln(self.encoder.prepare_tokens(x))
        for block in self.encoder.blocks:h=block(h)
        return self.head(self.encoder.final_norm(h).mean(1))

class MatchedTransformer(nn.Module):
    def __init__(self,ff):
        super().__init__();self.position=nn.Parameter(torch.zeros(1,33,128))
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(128,4,dim_feedforward=ff,dropout=0.,activation='gelu',batch_first=True,norm_first=True) for _ in range(4)])
        self.norm=nn.LayerNorm(128)
    def forward(self,x):
        h=x+self.position
        mask=torch.ones(x.shape[1],x.shape[1],device=x.device,dtype=torch.bool).triu(1)
        for layer in self.layers:h=layer(h,src_mask=mask,is_causal=True)
        return self.norm(h)

class RFClassifier(nn.Module):
    def __init__(self,backbone,hidden,classes,head_seed,is_qwen=True):
        super().__init__();self.backbone=backbone;self.is_qwen=is_qwen
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(head_seed)
            self.project=nn.Linear(64,hidden);self.cls=nn.Parameter(torch.randn(1,1,hidden)*.02);self.head=nn.Linear(hidden,classes)
    def embed(self,x):
        if x.ndim!=3 or x.shape[1:]!=(32,64):raise ValueError('RF input must be [B,32,64]')
        projected=self.project(x)
        return torch.cat([projected,self.cls.to(projected.dtype).expand(len(x),-1,-1)],1)
    def hidden(self,x):
        e=self.embed(x)
        if self.is_qwen:
            return self.backbone(inputs_embeds=e,attention_mask=torch.ones(e.shape[:2],dtype=torch.long,device=e.device),use_cache=False,return_dict=True).last_hidden_state
        return self.backbone(e)
    def forward(self,x):return self.head(self.hidden(x)[:,-1])

def small_match():
    torch.manual_seed(16001);q=Qwen3Model(tiny_config());drop_vocabulary(q)
    target=sum(p.numel() for p in q.parameters())
    # Projection/CLS/head are identical; match backbone counts without data.
    candidates=[dict(ff=ff,params=4*(4*128*128+4*128+4*128+257*ff+128)+256+33*128) for ff in range(64,1025,8)]
    best=min(candidates,key=lambda r:(abs(r['params']-target),r['ff']))
    native=MatchedTransformer(best['ff']);actual=sum(p.numel() for p in native.parameters())
    if actual!=best['params']:raise ValueError('Transformer parameter formula mismatch')
    if abs(actual/target-1)>.02:raise ValueError('Cannot match small architecture within2%')
    return dict(qwen_params=target,selected=best,candidates=candidates,relative_error=abs(actual/target-1),causal_both=True,cls_last_both=True)

def build(arm,seed,classes=10,engineering_config=None):
    torch.manual_seed(seed);head_seed=stable_seed('S16-common-head',seed,classes)
    if arm.startswith('R0'):
        m=ClassicTail(arm,classes);m.audit=dict(backbone='historical-style fresh tail',vocabulary_params=0)
        return m
    loading={}
    if arm=='R1_qwen_small':
        config=tiny_config();config._attn_implementation='sdpa';backbone=Qwen3Model(config);audit=drop_vocabulary(backbone);hidden=128
    elif arm=='R2_transformer':
        match=read(HERE/'small_architecture_match.json');backbone=MatchedTransformer(match['selected']['ff']);audit=dict(parameter_match=match['relative_error'],vocabulary_params=0);hidden=128
    else:
        manifest=read(HERE/'MODEL_MANIFEST.json')
        config=Qwen3Config.from_pretrained(manifest['local_path'],local_files_only=True);config.use_cache=False;config._attn_implementation='sdpa'
        if engineering_config is not None:config=engineering_config;config.use_cache=False;config._attn_implementation='sdpa'
        pretrained=arm in ['R3_pretrained_frozen','R5_pretrained_lora'] and engineering_config is None
        if pretrained:
            backbone,loading=Qwen3Model.from_pretrained(manifest['local_path'],config=config,torch_dtype=torch.bfloat16,
                use_safetensors=True,local_files_only=True,attn_implementation='sdpa',output_loading_info=True)
            if loading['missing_keys'] or loading['mismatched_keys']:raise ValueError('Incomplete official backbone weight load')
            if any(k!='lm_head.weight' for k in loading['unexpected_keys']):raise ValueError('Unexpected official checkpoint tensor')
        else:
            backbone=Qwen3Model(config).to(dtype=torch.bfloat16 if engineering_config is None else torch.float32)
        audit=drop_vocabulary(backbone);hidden=config.hidden_size
        backbone.requires_grad_(False)
        if arm in ['R5_pretrained_lora','R6_random_lora']:
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(stable_seed('S16-common-LoRA',seed))
                backbone=get_peft_model(backbone,LoraConfig(r=8,lora_alpha=16,lora_dropout=0.,target_modules=['q_proj','v_proj'],bias='none',task_type=TaskType.FEATURE_EXTRACTION))
        audit.update(pretrained=pretrained,revision=manifest['revision'],loading=loading)
    m=RFClassifier(backbone,hidden,classes,head_seed,is_qwen=arm!='R2_transformer');m.audit=audit
    return m

def parameter_audit(model):
    total=sum(p.numel() for p in model.parameters());trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)
    return dict(total_active_tail_params=total,trainable_tail_params=trainable,unused_word_embedding_params=0,lm_head_params=0,details=model.audit)

def trainable_state(model):return {n:p.detach().cpu() for n,p in model.named_parameters() if p.requires_grad}
def load_trainable(model,state):
    required={n for n,p in model.named_parameters() if p.requires_grad}
    if set(state)!=required:raise ValueError('Trainable checkpoint key mismatch')
    with torch.no_grad():
        for n,p in model.named_parameters():
            if n in state:p.copy_(state[n].to(p.device))
