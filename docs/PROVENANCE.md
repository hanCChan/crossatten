# 代码与数据来源

## 论文 1 模型代码（未修改）

`lib/legacy_rfhstu/` 由 `/workspace/gai/branches/shen-p0-p1/src/rfhstu/`（commit eb4ce2c0）原样复制而来，这些文件与用户公开仓库 keyan 的 `src/rfhstu` 同源。
本次复制时联网 clone keyan 被权限拦截，因此无法与 keyan@e91eb1d 逐字节比对。已知的是：参数量与论文 1 表 III 完全一致（CrossAttn-OOB 625,896，TrueIB 580,008）。

| 文件 | SHA256 |
|---|---|
| cnn_stem.py | 137f5ad37f90599f6378d8614fab9911dad07d9def0cbda3071446f7828f24bb |
| features.py | c994e56497bebf015a98b469469ecb8838bbae6e9be26aa43ba95f0ba19a933d |
| losses.py | d77eeedd3c74d1e387ef2ecf0efebc1f567105b238fe9fc4158a22e6d0cef186 |
| mixstyle.py | c23de1df2ff3b0cfad250323781f7d97fe04781e0f00c0ed9bb0772f9bf66bf7 |
| models.py | 28d865b46f8072d43778f440e51641b7b9b39c17d44820e418ab3d5f7fee3b0b |
| multiscale.py | e0888303717d611dbd9d84c272cfaa739d1e5765f9014640bc95a0464d2904ca |
| oob_fusion.py | 06a2da3d660cc31dabc67305552cc5cad24a63231ca8a84ac7de1d26d351b1ab |

论文 1 使用的训练脚本 `experiments/paper1_audit/scripts/run_final_short.py` 本地没有。阶段 01 的 `p1_fixed` 配方是按论文 1 表 IV 和用户方案文档复现的：AdamW 1e-3、wd 5e-4、batch 64、第 4 个 epoch。

## Shen et al. 的方法（重新实现，非复制）

- 来源：github.com/gxhen/receiverAgnosticRFFI（CC BY-NC-SA 4.0）。本地参考副本在 `/workspace/gai/branches/p2-rx-null/external/receiverAgnosticRFFI/`。
- `lib/xa/frontends.py::shen_cis` 是对 `ChannelIndSpectrogram` 的 PyTorch 移植，以 float64 计算。`tests/test_core.py` 将它与原 numpy/scipy 逻辑逐元素比较，最大误差小于 1e-5（实际为 2e-7，来自 float32 输出舍入）。
- `lib/xa/models.py::ShenCISPipeline` 移植自 `non_adv_net`：Keras 的 glorot 初始化、'same' 填充（k7/s2 时为 2/3 非对称填充）、channels-last 展平顺序。参数量为 13,442,954。
- AWGN 增强与训练配方（SGD 1e-3，动量 0.9，ReduceLROnPlateau(0.2, 10)，EarlyStopping(20)）来自 `dataset_preparation.awgn` 和 `main.py`。
- 与原实现的差异：
  - 原实现每个 batch 都随机抽样；这里改为按 epoch 打乱后不放回遍历。
  - 原实现每个 epoch 的验证噪声都重新生成；这里固定使用一次噪声实现，以保证可复现。
  - 源验证集按本协议取 40 包/TX。
- 使用说明：此为非商业学术用途；论文中需引用 Shen et al. TMC2024，并写明"PyTorch 重新实现"。

## 数据

- 原始 HDF5 在 `/workspace/gai/data/external/shen_receiver_agnostic`（软链接为 `data/raw`）。
- 10 台 RX 训练文件的 SHA256 与 gai g1 登记一致。4 台确认 RX 的 SHA256 已记录在 `00-signal-audit/out/00-1-cache-manifest.json`。
- `data/cache` 与 rl1 g1_cache_v1 逐字节一致（10 台 RX 的 I/Q、行号、标签）。

## 可复现性说明

- 每次运行设置 `torch.use_deterministic_algorithms(True, warn_only=True)`，并固定数据顺序和 AWGN 的随机数生成器。
- 但 `nn.MultiheadAttention` 在 CUDA 上的反向传播（memory-efficient attention）不是确定性的，日志中有相应警告。因此 CrossAttn 和 TF 的运行无法做到比特级复现，只能做到统计意义上的复现。Shen-CIS 不含注意力层，不受此影响。
