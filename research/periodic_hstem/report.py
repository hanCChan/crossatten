"""Generate a figure and Chinese report from frozen derived metrics only."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE=Path(__file__).resolve().parent


def main():
    r=json.loads((HERE/"AUDIT_RESULT.json").read_text(encoding="utf-8"))
    q=json.loads((HERE/"VERIFICATION.json").read_text(encoding="utf-8"))
    reg=json.loads((HERE/"AUDIT_REGISTRATION.json").read_text(encoding="utf-8"))
    if q["status"]!="PASS":raise ValueError("Verification required")
    with np.load(HERE/"PACKET_METRICS.npz",allow_pickle=False) as f:
        rho=f["rho20_by_offset"][:,0]
        adjacent=f["adjacent_min_by_offset"].min(1)
        sliding=f["sliding80_rho20"]
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,
                         "svg.hashsalt":"wisig-periodic-feasibility-v1"})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout="constrained")
    ax=axes[0,0]
    ax.plot(np.arange(1,81),r["lag_spectrum_median"],color="#176b87",lw=1.8)
    for lag in (20,40,60,80):ax.axvline(lag,color="#d78922",alpha=.35,lw=1)
    ax.set(xlabel="Lag (samples)",ylabel="Median normalized correlation",title="A. Whole-packet lag spectrum (descriptive)",ylim=(0,1.03))
    ax=axes[0,1]
    starts=r["sliding80_starts"]
    p10,p50,p90=np.quantile(sliding,[.1,.5,.9],axis=0)
    ax.fill_between(starts,p10,p90,alpha=.18,color="#176b87",label="10th-90th percentiles")
    ax.plot(starts,p50,color="#176b87",label="Median")
    ax.set(xlabel="80-sample window start",ylabel="Lag-20 correlation",title="B. Repetition changes within the 256-point crop",ylim=(0,1.03))
    ax.legend(loc="lower left",fontsize=9)
    ax=axes[1,0]
    for values,label,color in ((rho,"Primary window [20,140)","#176b87"),(adjacent,"Worst adjacent pair / 20 offsets","#d78922")):
        ax.plot(np.sort(values),np.arange(1,len(values)+1)/len(values),label=label,color=color)
    ax.axvline(.7,color="#b43b3b",ls="--",lw=1)
    ax.set(xlabel="Normalized correlation",ylabel="Empirical cumulative fraction",title="C. All sampled packets retained",xlim=(0,1.01),ylim=(0,1.01))
    ax.legend(loc="upper left",fontsize=8)
    ax=axes[1,1]
    rows=r["rx_day"]
    values=np.array([a["pass_fraction"]*100 for a in rows])
    ax.barh(np.arange(len(rows)),values,color=["#b43b3b" if v<80 else "#176b87" for v in values])
    ax.set_yticks(np.arange(len(rows)),[f"{a['rx']} D{a['day']} (n={a['packets']})" for a in rows],fontsize=8)
    ax.axvline(80,color="#b43b3b",ls="--",lw=1)
    ax.set(xlabel="Periodic feasibility pass rate (%)",title="D. One RX/day misses the registered 80% gate",xlim=(0,108))
    ax.invert_yaxis()
    for i,v in enumerate(values):ax.text(v+.6,i,f"{v:.2f}",va="center",fontsize=8)
    fig.suptitle("WiSig source-only periodic feasibility | 1,872 packets | 0 classifier fits",fontsize=14)
    fig.savefig(HERE/"FEASIBILITY.png",dpi=170)
    svg=HERE/"FEASIBILITY.svg"
    fig.savefig(svg,metadata={"Date":None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines())+"\n",
                   encoding="utf-8",newline="\n")
    plt.close(fig)
    lines=["# WiSig 周期 HStem：有界源包核验结果", "",
        "**20点重复结构得到支持；预注册的固定窗口方案未通过局部门，三结构训练未启动。** 这不是二维HStem识别性能的负结果，也不能解释成方向A已被否定。", "",
        f"检查前提交 `{q['preregistration_commit'][:7]}`，沿用C的源域白名单与原train_ids，只检查两视图共同fit的6台RX、日期1–2，共{r['packets']:,}包；每实际存在RX×TX×day抽16包，共{len(r['cells'])}格。3个缺失格原样报告。checkpoint_val/source_selection没有IQ数值读取，historical held6/target8/日期3–4没有访问。完整源IQ文件为核对SHA读取了原始字节，只有抽中行被复制并作数值计算；不把哈希读取描述成只接触抽中包的文件字节。", "",
        "| 预注册检查 | 结果 |", "|---|---:|",
        f"| 总体通过率，要求≥90% | {r['passed_packets']}/{r['packets']} = {r['pass_fraction']*100:.4f}% |",
        "| 最差RX×day通过率，要求≥80% | 8-7，D2：126/160 = 78.75%，未通过 |",
        f"| 主要窗口[20,140) lag20相关，中位数 / p10 | {r['rho20_primary']['median']:.6f} / {r['rho20_primary']['p10']:.6f} |",
        f"| 全部窗口起点下最差相邻周期相关，中位数 / p10 | {r['worst_adjacent_coherence']['median']:.6f} / {r['worst_adjacent_coherence']['p10']:.6f} |",
        f"| 尾段[160,256) lag20相关中位数 | {r['region_rho20']['160:256']['median']:.6f} |",
        f"| 整包[0,256) lag20相关中位数 | {r['region_rho20']['0:256']['median']:.6f} |",
        "| 分类器fit / TX判别或RX泛化评分 | 0 / 未执行 |", "",
        "![源域周期核验](FEASIBILITY.png)", "",
        "## 证据怎样解释", "",
        "采样率25 Msps与L-STF的0.8 μs周期给出20点候选。局部窗口的高相关及整包lag20/40/60/80峰支持重复波形确实存在；滑窗结果又显示后段重复性下降，因此把全部256点直接当成同质短符号矩阵缺少支持。相关下降本身不精确定位L-STF边界，也不能区分边界、检测误差、干扰、信噪比或硬件响应的贡献。", "",
        "主要候选是[20,140)六行二十列；为避免一次整体相关掩盖局部失配，门要求全部20种起点偏移下的相邻行相关≥0.7，并要求lag20相对lag19/21的优势≥0.05。后一个条件防止把单音或很平滑的波形误当成20点特有周期。它检验周期证据是否足够明确，不等同于判断包是否为有效Wi-Fi。", "",
        f"相邻周期相关门有{q['failure_counts']['coherence']}包未过，周期区分度门有{q['failure_counts']['neighbor_margin']}包未过，重叠{q['failure_counts']['both']}包，合计{q['failure_counts']['either']}包；未删除任何失败包。", "",
        "局部门仅在8-7的D2未过：126包通过，80%门要求128包，相差2包。**这是有限源样本上的工程停止条件，不能把1.25个百分点的差距解释成统计上证实固定窗口不可用，更不能推出周期建模不可能有效。** 高相关也不识别绝对短符号起点；p列目前是包相对坐标，不保证跨包处于同一物理相位。", "",
        "## 本轮决定", "",
        f"机器判定：`{r['status']}`。遵守检查前冻结的停止规则，不在看到数据后改用[40,160)、放宽门槛、剔除失败包、追加样本或训练种子。本轮没有实现并训练二维候选，也没有将lag20相位通道重新包装成新方法。", "",
        "三结构竞争保留为条件式方案：频域HStem+MHA、参数受控的一维时域HStem、周期二维HStem；训练之前仍须独立冻结输入信息、整包保留方式、参数容差与共同分类器初始化。当前可行性结果不支持跳过这一步。已有EMA平均收益仍需在之后的独立强基线比较中保留，不把普通ERM的结构竞争当作超过最强标准配方的证明。", "",
        "要继续这一物理假设，缺的是可信边界或能显式处理起点不确定性的表示及公平对照，而不是另一个Loss。那将是新的、独立预注册的假设；本轮不自动添加自适应对齐、有效包筛选、均值/残差分支或其他补救结构。正式CRD与新RX确认权限保持原资格限制。", "",
        "## 可复核材料", "",
        f"合成测试9项通过；`verify.py`从逐包派生量和原始元数据独立重算抽样、决策及每个分组，{q['check_count']}项核验通过；核验不读取IQ。输出PNG及SVG已生成。数值证据、源路径、时间与SHA保留如下：", "",
        "- [检查前登记](AUDIT_REGISTRATION.json) 与 [方法和停止规则](README.zh-CN.md)。",
        "- [全部逐包派生量](PACKET_METRICS.npz)、[结果JSON](AUDIT_RESULT.json) 与 [访问记录](ACCESS_LOG.json)。",
        "- [复核结果](VERIFICATION.json)、[复核代码](verify.py) 与 [图表SVG](FEASIBILITY.svg)。", "",
        "外部物理依据：[WiSig原论文 §III–IV](https://arxiv.org/html/2112.15363)、[MathWorks L-STF文档](https://www.mathworks.com/help/wlan/ref/wlanlstf.html)。没有本轮新增的目标域性能、LoRa迁移结果或成功机制声明。", ""]
    (HERE/"RESULTS.zh-CN.md").write_text("\n".join(lines),encoding="utf-8",newline="\n")
    print(json.dumps(dict(status="REPORT_AND_FIGURES_CREATED",scientific_fits=0)))


if __name__=="__main__":main()
