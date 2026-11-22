"""User-requested A/B/C extension: all H0/P0/S1/H1, frozen normalization."""
import csv
import hashlib
from pathlib import Path
import sys

import h5py
import numpy as np
from scipy.stats import spearmanr
import torch

from geometry_stats import geometry
from s25_common import HERE, ROOT, P, authorize_source, committed, digest, now, read, source_validation, validate, write
from source_geometry import extract, csv_write


def validate_extension():
    validate()
    path=HERE/'analysis_extension_001.json';committed(path);r=read(path)
    assert r['parent_registration_sha256']==digest(HERE/'registration.json')
    assert r['initial_source_result_sha256']==digest(HERE/'SOURCE_GEOMETRY.json')
    assert r['new_fits']==0
    for rel,expected in r['code_sha256'].items():
        if digest(ROOT/rel)!=expected:raise PermissionError('Extension code changed: '+rel)
    for rel,expected in r['evidence_sha256'].items():
        if digest(ROOT/rel)!=expected:raise PermissionError('Historical evidence changed: '+rel)
    return r


def rho(x,y):
    x,y=np.asarray(x,dtype=float),np.asarray(y,dtype=float)
    good=np.isfinite(x)&np.isfinite(y);x,y=x[good],y[good]
    if len(x)<3 or np.ptp(x)<=1e-12 or np.ptp(y)<=1e-12:return None
    return float(spearmanr(x,y).statistic)


def centered(values,tx,rx):
    result=np.asarray(values,dtype=float).copy()
    for t in np.unique(tx):
        for r in np.unique(rx):
            ids=(tx==t)&(rx==r)&np.isfinite(result)
            if ids.any():result[ids]-=result[ids].mean()
    return result


def source_metadata(fold,rx,rows,y):
    authorize_source(fold,rx)
    path=Path('D:/crossatten/stage_1/data/raw')/(rx+'_train.h5')
    rows=np.asarray(rows,dtype=np.int64);order=np.argsort(rows);inverse=np.argsort(order)
    output={}
    with h5py.File(path,'r') as f:
        for key in ['CFO','SNR','label']:
            dataset=f[key]
            if dataset.shape==(1,8000):value=dataset[0,rows[order]][inverse]
            elif dataset.shape==(8000,1):value=dataset[rows[order],0][inverse]
            else:raise ValueError('Unregistered metadata layout: '+str(dataset.shape))
            output[key]=np.asarray(value,dtype=np.float64)
    np.testing.assert_array_equal(output.pop('label'),np.asarray(y)+31)
    event=dict(time=now(),role='source_metadata',fold=fold,rx=rx,packets=len(rows),
               fields=['CFO','SNR','label'],waveform_field_read=False,
               selected_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),extension_sha256=digest(HERE/'analysis_extension_001.json'))
    with (HERE/'access.jsonl').open('a',encoding='utf-8',newline='\n') as stream:stream.write(__import__('json').dumps(event)+'\n')
    return output


def main():
    registration=validate_extension()
    torch.set_num_threads(2);torch.manual_seed(25025)
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    sys.path.insert(0,str(ROOT/'22-mechanism-closure'))
    from s22_models import build
    parent=read(HERE/'registration.json')
    all_models=parent['models']+registration['additional_models']
    for fold in (0,1):
        data=source_validation(fold)
        metadata={'CFO':np.empty(2000),'SNR':np.empty(2000)}
        for ri,rx in enumerate(P['folds'][fold]['source']):
            ids=data['rx']==ri
            values=source_metadata(fold,rx,data['rows'][ids],data['y'][ids])
            for key in metadata:metadata[key][ids]=values[key]
        np.savez(HERE/'features'/f'metadata_f{fold}.npz',**metadata,y=data['y'],rx=data['rx'],rows=data['rows'])
        for record in [r for r in registration['additional_models'] if r['fold']==fold]:
            assert digest(Path(record['checkpoint']))==record['checkpoint_sha256']
            model=build(record['arm'],record['data_seed']).cuda().eval()
            model.load_state_dict(torch.load(record['checkpoint'],map_location='cuda',weights_only=True),strict=True)
            for p in model.parameters():p.requires_grad_(False)
            features=extract(model,data['x'])
            np.savez(HERE/'features'/(record['id']+'.npz'),**features,y=data['y'],rx=data['rx'],rows=data['rows'])
            print(record['id'],'source_acc',float(np.mean(features['logits'].argmax(1)==data['y'])),flush=True)
            del model;torch.cuda.empty_cache()
    metrics=[];margins=[];conditions=[];correlations=[];joined=[]
    bands=list(csv.DictReader((ROOT/'23-hstem-mechanism-evidence/kernel_bands.csv').open(encoding='utf-8-sig')))
    for record in all_models:
        with np.load(HERE/'features'/(record['id']+'.npz'),allow_pickle=False) as data:
            tx,rx=data['y'].copy(),data['rx'].copy();logits=data['logits'].astype(float);final=data['final'].astype(float)
            for layer in ['tokens','pre_mha','final']:
                row=dict(arm=record['arm'],fold=record['fold'],seed=record['seed'],layer=layer,**geometry(data[layer],tx,rx));metrics.append(row)
                band=next((b for b in bands if b['arm']==record['arm'] and int(b['fold'])==record['fold'] and int(b['seed'])==record['seed']),None)
                if band is not None:joined.append(dict(**row,kernel_high_energy=float(band['high_energy']),kernel_size=int(band['kernel_size'])))
        others=logits.copy();others[np.arange(len(tx)),tx]=-np.inf
        margin=logits[np.arange(len(tx)),tx]-others.max(1)
        weights=torch.load(record['checkpoint'],map_location='cpu',weights_only=True)['net.classifier.weight'].double().numpy()
        normalized=final/np.linalg.norm(final,axis=1,keepdims=True)
        cosine=normalized@(weights/np.linalg.norm(weights,axis=1,keepdims=True)).T
        wrong=cosine.copy();wrong[np.arange(len(tx)),tx]=-np.inf
        cosine_margin=cosine[np.arange(len(tx)),tx]-wrong.max(1)
        predictions=logits.argmax(1)
        meta=np.load(HERE/'features'/f"metadata_f{record['fold']}.npz",allow_pickle=False)
        np.testing.assert_array_equal(meta['y'],tx);np.testing.assert_array_equal(meta['rx'],rx)
        for ri,name in enumerate(P['folds'][record['fold']]['source']):
            for t in range(10):
                ids=(rx==ri)&(tx==t)
                margins.append(dict(arm=record['arm'],fold=record['fold'],seed=record['seed'],rx=name,tx_raw_id=t+31,
                    accuracy_pct=float(np.mean(predictions[ids]==tx[ids])*100),raw_margin=float(margin[ids].mean()),
                    cosine_weight_margin=float(cosine_margin[ids].mean()),packets=int(ids.sum())))
            ids=rx==ri
            centroids=[normalized[tx==t].mean(0) for t in range(10)]
            dispersion=float(np.mean([np.sum((normalized[ids&(tx==t)].mean(0)-centroids[t])**2) for t in range(10)]))
            conditions.append(dict(arm=record['arm'],fold=record['fold'],seed=record['seed'],rx=name,
                same_tx_rx_centroid_dispersion=dispersion,raw_margin=float(margin[ids].mean()),
                cosine_weight_margin=float(cosine_margin[ids].mean()),source_accuracy_pct=float(np.mean(predictions[ids]==tx[ids])*100)))
        for field in ['SNR','CFO']:
            value=meta[field].astype(float)
            if field=='CFO':value=np.abs(value)
            residual=centered(value,tx,rx)
            correlations.append(dict(arm=record['arm'],fold=record['fold'],seed=record['seed'],field=field,
                transformation='as_reported' if field=='SNR' else 'absolute_as_reported',
                raw_rho_margin=rho(value,margin),within_tx_rx_centered_rho_margin=rho(residual,centered(margin,tx,rx)),
                raw_rho_cosine_margin=rho(value,cosine_margin),within_tx_rx_centered_rho_cosine_margin=rho(residual,centered(cosine_margin,tx,rx)),
                finite_packets=int(np.isfinite(value).sum()),within_cell_residual_range=float(np.ptp(residual[np.isfinite(residual)]))))
        meta.close()
    associations=[]
    for layer in ['tokens','pre_mha','final']:
        rows=[r for r in joined if r['layer']==layer]
        for metric in ['conditional_rx','tx_to_rx_centroid_ratio','tx_to_within_class_ratio']:
            associations.append(dict(layer=layer,metric=metric,models=len(rows),rho=rho([r['kernel_high_energy'] for r in rows],[r[metric] for r in rows]),
                scope='Descriptive across18 fitted models; arm, shared cohorts and optimization confound correlation; no p-value or causal claim.'))
    legacy=list(csv.DictReader((ROOT/'22-mechanism-closure/tx_rx_metrics.csv').open(encoding='utf-8-sig')))
    legacy=[r for r in legacy if r['role']=='dev_val']
    tx_comparisons=[]
    for baseline in ['P0','S1','H1']:
        for row in [r for r in legacy if r['arm']==baseline]:
            h=next(r for r in legacy if r['arm']=='H0' and all(r[k]==row[k] for k in ['fold','seed','rx','tx']))
            tx_comparisons.append(dict(cohort='S22_archival_GPU',baseline=baseline,fold=int(row['fold']),seed=int(row['seed']),rx=row['rx'],tx_raw_id=int(row['tx'])+31,
                h0_accuracy_pct=100*(1-int(h['errors'])/int(h['packets'])),baseline_accuracy_pct=100*(1-int(row['errors'])/int(row['packets'])),
                h0_minus_baseline_pp=100*(int(row['errors'])-int(h['errors']))/int(row['packets']),
                h0_minus_baseline_margin=float(h['margin'])-float(row['margin'])))
    for filename,rows in [('mechanism_geometry.csv',metrics),('source_tx_rx_margins.csv',margins),('source_rx_conditions.csv',conditions),
                          ('metadata_correlations.csv',correlations),('kernel_geometry_join.csv',joined),('matched_tx_rx_comparisons.csv',tx_comparisons)]:csv_write(HERE/filename,rows)
    write(HERE/'MECHANISM_EXTENSION.json',dict(status='COMPLETE',time=now(),models=24,new_feature_predictions=24000,total_source_feature_predictions=48000,
        new_fits=0,new_optimizer_steps=0,new_dev_waveform_reads=0,new_confirm_blind_reads=0,geometry=metrics,
        metadata_correlations=correlations,kernel_geometry_associations=associations,
        prior_h0_p0_source_results_retained=True,extension_sha256=digest(HERE/'analysis_extension_001.json')))


if __name__=='__main__':main()
