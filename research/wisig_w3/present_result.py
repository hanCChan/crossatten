"""Render the fixed W3-B outcome for review; does not fit or select any model."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent


def main():
    result=json.loads((HERE/'B_RESULT.json').read_text(encoding='utf-8'));e=result['models']['EMA'];er=result['models']['ERM']
    methods=('identity','shift','diagonal','rank2','rank8','shuffled_rank8')
    labels=('Identity','RX shift','Diagonal','Rank 2','Rank 8','Shuffled rank 8')
    colors=['#8c9bad','#718da6','#527ea0','#527ea0','#bc493f','#b7b7b7']
    fig,axes=plt.subplots(1,2,figsize=(11.5,4.6),layout='constrained')
    errors=[e['method_means'][m]['transport_error'] for m in methods]
    ratios=[e['method_means'][m]['median_pair_context_TX_squared_distance_ratio'] for m in methods]
    for ax,values,title in zip(axes,(errors,ratios),('Held-TX transport error (lower is better)','TX geometry: median squared-distance ratio')):
        bars=ax.bar(np.arange(6),values,color=colors,width=.7)
        ax.set_xticks(np.arange(6),labels,rotation=25,ha='right');ax.set_title(title,fontsize=11)
        ax.set_ylim(0,max(values)*1.2);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        ax.spines[['top','right']].set_visible(False)
        for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+.015,f'{v:.3f}',ha='center',fontsize=9)
    axes[1].axhline(.9,color='#333333',ls='--',lw=1,label='Prespecified median gate = 0.9');axes[1].legend(frameon=False,fontsize=8,loc='upper right')
    fig.suptitle('W3-B / frozen EMA / 96 auxiliary TX / 6 source RX / D1-D2 swaps',fontsize=12)
    fig.savefig(HERE/'B_TRANSPORT_GEOMETRY.png',dpi=180);fig.savefig(HERE/'B_TRANSPORT_GEOMETRY.svg');plt.close(fig)
    rx=json.loads((HERE/'B_ROLES.json').read_text(encoding='utf-8'))['RX'];grid=np.full((6,6),np.nan)
    for s in range(6):
        for d in range(6):
            if s!=d:grid[s,d]=100*e['pair_relative_error_reductions'][rx[s]+'->'+rx[d]]
    fig,ax=plt.subplots(figsize=(7,6),layout='constrained');im=ax.imshow(grid,cmap='RdBu',vmin=-10,vmax=10)
    for s in range(6):
        for d in range(6):
            if s!=d:ax.text(d,s,f'{grid[s,d]:+.2f}%',ha='center',va='center',color='white' if abs(grid[s,d])>6 else '#222222',fontsize=10)
    ax.set_xticks(range(6),rx);ax.set_yticks(range(6),rx);ax.set_xlabel('Destination source RX');ax.set_ylabel('Origin source RX')
    ax.set_title('Rank 8 vs strongest simple control\n12 / 30 directed RX pairs improve; four contexts pooled',fontsize=12)
    fig.colorbar(im,ax=ax,label='Relative transport-error reduction (%)',shrink=.8)
    fig.savefig(HERE/'B_DIRECTED_RX_EFFECTS.png',dpi=180);fig.savefig(HERE/'B_DIRECTED_RX_EFFECTS.svg');plt.close(fig)
    table=[]
    for m in methods:
        q=e['method_means'][m]
        table.append(dict(model='EMA',method=m,transport_error=q['transport_error'],
            geometry_median=q['median_pair_context_TX_squared_distance_ratio'],geometry_minimum=q['minimum_pair_context_TX_squared_distance_ratio'],
            labeled_prototype_accuracy_percent=100*q['labeled_prototype_accuracy']))
    with (HERE/'B_METHOD_SUMMARY.csv').open('w',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=table[0]);w.writeheader();w.writerows(table)
    rows='\n'.join(f"| {labels[i]} | {v['transport_error']:.6f} | {v['geometry_median']:.3f} | {v['geometry_minimum']:.3f} | {v['labeled_prototype_accuracy_percent']:.3f}% |" for i,v in enumerate(table))
    contexts='\n'.join(f"| {c['id']} | {c['strongest_simple_control']} | {c['control_error']:.6f} | {c['rank8_error']:.6f} | {100*c['relative_error_reduction']:+.3f}% |" for c in e['contexts'])
    lower,upper=e['vs_strongest_simple_control']['relative_error_reduction_CI95']
    report=f'''# W3-A/B 结题：存在简单统计转移，rank8增量与几何保护假设未通过

**决策：`{result['status']}`。停止本轮预注册rank8实现；W3-C不启动。新增神经网络训练为0。**

这里停止的是“rank8比简单变化预测得更好，同时保持TX类间几何”这一具体假设，不能把结果扩展成“所有接收机相关变化都不能跨TX转移”。LoRa稿件不变，W2/P2/F2/M1的原STOP不变。

## 1. 最直接的结果

EMA的rank8迁移误差为 **{e['rank8_error']:.6f}**，最强简单对照为 **{e['strongest_simple_control_error']:.6f}**（四个上下文均为逐维仿射）。相对误差下降 **{100*e['relative_error_reduction']:+.3f}%**，即误差增加1.87%；TX聚类bootstrap的95%条件区间为 **[{100*lower:+.3f}%, {100*upper:+.3f}%]**，没有通过预设的至少15%改善门槛。

四上下文只有一个为正，30个有向RX对只有 **{e['positive_directed_RX_pairs']}/30** 改善（门槛20/30）。ERM同轨迹敏感性参照也为 **{100*er['relative_error_reduction']:+.3f}%**，不是EMA单独遇到的反例。

rank8的TX间距离平方保持率，中位数 **{e['median_TX_squared_distance_ratio']:.3f}**，最差RX对/上下文的中位数 **{e['minimum_TX_squared_distance_ratio']:.3f}**；均低于0.9/0.8预设门槛。0.741表示距离**平方**的保持率，不是准确率，也不能解释成欧氏距离恰好减少25.9%。

![迁移误差与类间几何](B_TRANSPORT_GEOMETRY.png)

## 2. W3-A 已通过的资产资格

官方ManyTx的150个TX中，先排除ManyRx10TX与ManySig6TX的并集11TX，留下139个候选。只按计数筛出在六个合法源RX、D1/D2上均至少32个raw包的 **96个辅助TX**。没有按模型表现筛选TX。

六个RX为 `13-7, 14-7, 18-19, 18-2, 8-14, 8-7`，全部属于原C V0的fit_RX；日期仅2021_03_01和2021_03_08。最终为 **1,152个单元、36,864个唯一包**。TX身份与原ManyRx/ManySig目标TX并集不重叠。

原始文件是单体压缩包，先不解码数值地存储，元信息索引使用seek跳过全部数值载荷。扫描21,600个数组、读取1,512,134字节元信息，与所有数值区间的读取交集为0。实际1152个入选单元的包数与官方Full计数逐一相符；只有封存B协议和白名单后才读取其raw前32行。目标RX、D3/D4、目标TX和EQ波形均未解码。

跨RX的相同行号不表示同一次空口发射；raw与EQ也不按索引配对。包键明确为(TX,RX,date,raw,prefix-row)。Full额外RX的历史暴露资格仍未知，合格的新盲测RX仍为0；本轮不能重置这个历史。

## 3. 冻结协议与执行量

读取IQ前，协议、94项代码/证据哈希、模型身份、TX角色和特征合成检验已提交到 **`5165f11`**。第一次正式IQ特征读取记录见[B_FEATURE_ATTEMPT.json](B_FEATURE_ATTEMPT.json)，固定结果随后提交到`e6a92f2`。本报告属于结果解释，没有改变候选、超参数或门槛。

使用原C V0/S0、wd=.001的EMA selected作主读出，同轨迹ERM selected作敏感性参照。特征为final_norm后池化的64维；没有用原10类分类头给96个辅助TX评分。两个模型的权重与缓冲区读取前后精确哈希相同，所有模块保持eval，无梯度、无BN适配。此处是固定的一条训练轨迹，不能推广成所有骨干与种子的规律。

96TX按固定SHA256排序切成48/48，两个方向互换；再与D1拟合→D2检验、D2拟合→D1检验组合，得到四上下文。每个上下文的PCA仅用fit-TX、拟合日、六源RX的288个均值；没有held-TX或检验日拟合。30个有向RX对全部报告。

六方法为恒等、RX均值偏移、逐维仿射、rank2残差、预指定rank8残差、打乱fit-TX对应的同容量rank8。固定岭系数0.1×设计Gram平均对角值（下限1e-12），不搜索超参数。shuffled使用无固定点的置换。rank8本身是普通低秩回归，不作为原创算法。

**新增神经拟合0；浅层变换拟合1,200；PCA拟合8；恒等评估240。** 增加了明确披露的辅助观测及浅层拟合数据预算；两个冻结模型各读出同一36,864包，没有神经模型训练或目标端适配。ERM是同一训练轨迹的参照，不是独立训练重复。

主量是检验日held-TX均值的归一化平方误差。各上下文从恒等/偏移/逐维/rank2里取held误差最低的对照，是事前声明的诊断oracle参照，不是部署时可用的选模器；它不参与任何重拟合。完整控制结果如下。

| EMA方法 | 迁移误差↓ | 距离平方保持率中位数 | 最差保持率 | 有标签原型诊断Accuracy |
|---|---:|---:|---:|---:|
{rows}

**最后一列只用于判别几何诊断：** 使用目标RX的held-TX前16包作为有标签原型，源RX同TX后16包为query，48类最近原型识别。它使用了检验端标签，绝不是可部署的未知RX Accuracy，也不能与原ManyRx/ManySig十类/六类排行榜相比较。rank8相对主对照为{e['labeled_prototype_accuracy_delta_pp']:+.3f}pp，平均归一化margin变化{e['normalized_margin_delta']:+.6f}，这两项通过，但不能抵消迁移误差与几何门槛失败。

## 4. 四划分与所有RX对

| 上下文 | 最强简单对照 | 对照误差 | rank8误差 | 相对误差下降 |
|---|---|---:|---:|---:|
{contexts}

![全部有向RX对](B_DIRECTED_RX_EFFECTS.png)

区间按96个held-TX聚类：先平均各TX的全部RX对及两个检验日期，再做20,000次seed31013重采样。不能把重复RX对、方法或包行当作独立重复。区间条件于这套固定提取器、划分和六源RX；没有涵盖新接收机、训练重启、拟合集变化或独立采集不确定性。共享的fit集合与交叉拟合依赖也限制了总体统计推断。

## 5. 保留什么，否定什么

保留两个描述性发现：普通均值偏移已经降低了跨TX、跨源日期的迁移误差，且距离平方保持率为1；正确对应rank8也明显优于同容量打乱对照（误差相对下降{100*e['vs_shuffled_rank8']['relative_error_reduction']:.2f}%）。这支持存在可预测的统计成分，不能直接识别为接收机硬件响应。

否定本轮增量主张：rank8未超过更简单的逐维校准，且类间几何压缩明显。逐维校准本身也压缩类间几何，不能因为它在迁移误差上最好就将其晋升为新增强算法。rank2也不事后替换为候选。**接近目标RX分布、优于被破坏对应关系的对照，都不足以证明能安全提高TX识别。**

接收机、位置、信道、TX状态与采集日期在现有观测里相互混杂。对一个RX对拟合出的Phi应称“接收机相关统计变换”，不能称独立识别的硬件响应。辅助TX数量增加也是数据预算变化，不能暗中并入“同数据领先”的叙述。

## 6. 原创性与下一步决策

[GAN-RXA原文](https://arxiv.org/html/2303.14312v2)已经使用辅助TX/RX学习接收变化，再迁移到新TX；“加入更多TX预训练”与“学习RX增强”本身不足以构成新颖性。本轮没有提出新神经算法、未知RX正向结果或90%成果。

W3-C保持未授权，且本轮机制门槛失败，不具备自动进入训练的依据。不改rank、岭参数、TX划分、包数或增加生成器补救。若将来重开，必须先提供能区别已有RX增强、具有额外观测验证条件的新机制。本轮结果适合保留为机制边界/否定性证据，不能写成成功方法贡献。此前完成的[LoRa投稿准备](../../publication/lora_hstem/README.md)继续作为已经成立的成果。

## 7. 审阅入口与来源

- [预注册](PREREGISTRATION_B.zh-CN.md)、[机器注册](B_EXECUTION_REGISTRATION.json)、[数据白名单](B_SOURCE_MANIFEST.json)、[完整TX角色](B_ROLES.json)。
- [主结果JSON](B_RESULT.json)、[逐held-TX指标69,120行](B_HELD_TX_METRICS.csv)、[逐RX对几何1,440行](B_PAIR_GEOMETRY.csv)、[方法汇总](B_METHOD_SUMMARY.csv)。这些是重复诊断行，不是69,120个独立样本。
- [IQ访问日志](B_IQ_ACCESS_LOG.json)、[冻结权重证明](B_FEATURE_RECEIPT.json)、[独立审计](B_INDEPENDENT_AUDIT_QA.json)。审计只读取特征和固定映射，不读IQ、不加载torch、不重新拟合。
- 原始波形不进入Git或审阅包。审阅包保留冻结特征、全部浅层参数、预注册代码与numpy审计入口，可在另一目录复算；该包不等于从原始IQ完整复跑。
- [UCLA官方数据说明](https://cores.ee.ucla.edu/downloads/datasets/wisig/)、[固定版本子集创建源码](https://github.com/WiSig-dataset/wisig-subset-creation/blob/bc8c6bd352a8c184a571e6b4c2d7a890cdfff3d8/py/002_FullWiSig_create_compact_datasets.py)。压缩包SHA256为`a8fc3e35134a240bfb4dab8862a6e482cef44de000b813d42417b853c47ccc7e`，PKL为`c0319174d40eb64bc49f201743941ebedc5cc0ced284c655cab798b2bdd44275`。

数据衍生特征仅供本研究审阅；原始数据按官方CC BY-NC-SA 4.0条件使用。下载与索引阶段的状态JSON是当时的审计快照，最终状态以B_RESULT和本报告为准。
'''
    (HERE/'README.zh-CN.md').write_text(report,encoding='utf-8',newline='\n')
    print('Wrote report, summary CSV and two publication-resolution figures')


if __name__=='__main__':main()
