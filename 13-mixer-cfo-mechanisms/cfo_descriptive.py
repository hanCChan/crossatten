"""Post-hoc source-only CFO description; never imported by training or gates."""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core import HERE, read, write, sha, now


def main():
    path=HERE/'source_cfo_packets.csv'
    df=pd.read_csv(path)
    good=df[df.coherence>=.7].copy()
    cells=good.groupby(['dataset','fold','tx','rx','day'],as_index=False).estimated_hz.median()
    rows=[];extremes=[];fig,axes=plt.subplots(2,1,figsize=(16,8),layout='constrained')
    for dataset in ['ManyRx','ManySig']:
        for fold in range(4):
            subset=cells[(cells.dataset==dataset)&(cells.fold==fold)].copy()
            tx=subset.groupby('tx').estimated_hz.median()
            subset['tx_centered_hz']=subset.estimated_hz-subset.tx.map(tx)
            rxday=subset.groupby(['rx','day']).tx_centered_hz.median()
            for row in subset[np.abs(subset.tx_centered_hz)>50000].to_dict('records'):extremes.append(row)
            reliability=df[(df.dataset==dataset)&(df.fold==fold)].assign(ok=lambda x:x.coherence>=.7).groupby('tx').ok.mean()
            rows.append(dict(dataset=dataset,fold=fold,reliable_cells=len(subset),
                tx_median_min_hz=float(tx.min()),tx_median_max_hz=float(tx.max()),
                tx_median_span_hz=float(tx.max()-tx.min()),
                tx_centered_rxday_span_hz=float(rxday.max()-rxday.min()),
                tx_centered_rxday_iqr_hz=float(rxday.quantile(.75)-rxday.quantile(.25)),
                cells_abs_tx_centered_over50khz=int((subset.tx_centered_hz.abs()>50000).sum()),
                minimum_tx_reliable_fraction=float(reliability.min()),maximum_tx_reliable_fraction=float(reliability.max())))
            if fold==0:
                subset['cell']=subset.rx.astype(str)+'/d'+subset.day.astype(str)
                table=subset.pivot(index='tx',columns='cell',values='tx_centered_hz')/1000
                ax=axes[0 if dataset=='ManyRx' else 1]
                vmax=max(1,float(np.nanquantile(np.abs(table.values),.99)))
                im=ax.imshow(table.values,aspect='auto',cmap='coolwarm',vmin=-vmax,vmax=vmax)
                ax.set_xticks(np.arange(len(table.columns)),table.columns,rotation=90,fontsize=7)
                ax.set_yticks(np.arange(len(table.index)),table.index);ax.set_ylabel('TX class')
                ax.set_title(dataset+': TX-centered observed CFO, fold0 source RX, days1-2')
                fig.colorbar(im,ax=ax,label='kHz (panel-specific scale)')
    fig.savefig(HERE/'P0_CFO_CONDITIONAL.png',dpi=150);plt.close(fig)
    pd.DataFrame(rows).to_csv(HERE/'cfo_descriptive.csv',index=False,lineterminator='\n',encoding='utf-8-sig')
    pd.DataFrame(extremes).to_csv(HERE/'cfo_extreme_cells.csv',index=False,lineterminator='\n',encoding='utf-8-sig')
    lines=['# CFO 源域补充描述性统计','',
        '本分析在看到CFO诊断输出后补充，不参与任何STOP/GO、频偏范围或预算选择。训练/门槛仍使用原预注册实现。',
        '只使用已有source_cfo_packets.csv，不重新读取IQ。每个TX/RX/day先取可靠包（coherence≥0.7）的CFO中位数，形成等权cell。',
        'TX中位数为跨RX/day的cell中位数；每个cell减去对应TX中位数，再按RX/day跨TX取中位数。该剩余量仍混合信道、交互与估计误差，不能叫作真实RX本振。','',
        '| 数据 | fold | TX中位数跨度 kHz | TX中心化后RX/day跨度 kHz | 其IQR kHz | 最低TX可靠比例 | 绝对剩余>50kHz的cell数 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in rows:lines.append(f"|{r['dataset']}|{r['fold']}|{r['tx_median_span_hz']/1000:.2f}|{r['tx_centered_rxday_span_hz']/1000:.2f}|{r['tx_centered_rxday_iqr_hz']/1000:.2f}|{r['minimum_tx_reliable_fraction']:.3f}|{r['cells_abs_tx_centered_over50khz']}|")
    lines+=['','![TX条件化观测频偏](P0_CFO_CONDITIONAL.png)','',
        '不同TX的观测CFO中位数存在关联，说明把每包总CFO归零会同时改变可能与TX身份有关的量。因此需要raw-only、corrected-only、raw+corrected以及双raw参数控制。',
        'ManyRx还出现少量TX条件化后超过50kHz的长尾cell（完整清单见cfo_extreme_cells.csv），图中的大范围色标由这些值主导。它们可能来自实际大频偏，也可能来自重复结构/对齐问题；本分析没有天然真值，不能判定。高相关系数和合成增量误差通过均不保证天然CFO绝对准确。未据此事后删除样本或修改门槛。',
        '此处没有天然CFO真值；不能从这个表推断天然分类错误由CFO造成，也不能声称已经唯一分离TX/RX频偏。']
    (HERE/'P0_CFO_SUPPLEMENT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8',newline='\n')
    write(HERE/'posthoc_cfo_manifest.json',dict(created_at=now(),role='DESCRIPTIVE_POST_HOC_NOT_USED_FOR_SELECTION',
        input_sha256=sha(path),code_sha256=sha(HERE/'cfo_descriptive.py'),rows=rows,modifies_training=False,modifies_gates=False))
    print('Post-hoc source CFO description complete. Training and gates unchanged.')


if __name__=='__main__':main()
