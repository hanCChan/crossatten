# S03/S06 权重归档

日期：2026-10-25。独立 Release：`paper-validation-2026-10-25`。
旧 `snapshot-2026-10-25` 的阶段00–02附件保持不变。

## 范围

新归档保存本地训练的阶段03及阶段06 `model.pt`，保留原仓库相对路径，包括阶段06 smoke。
正式准确率只引用通过审计的注册 fit，不把 smoke 计入模型成绩。
配置、训练曲线、指标、预测、数据访问和失败/中断记录在 Git 中；数据集、cache、凭据和第三方源码不在归档内。
源域决策及注册配置的冻结记录见[阶段06 README](../06-paper-validation/README.md)，结果见[中文结论](../06-paper-validation/summary/paper/CONCLUSION.md)。

## 下载与验证

在仓库根目录放置该 Release 的 `.part-*` 附件，执行：

```bash
sha256sum -c artifacts/PAPER_VALIDATION_SHA256SUMS
cat crossatten-paper-validation-2026-10-25.tar.zst.part-* | tar --zstd -xf -
sha256sum -c artifacts/paper-validation-checkpoints.sha256
```

附件按名称顺序拼接；清单列出每块 SHA256，解包后的清单核验每个权重。
归档数量、字节数、报告及附件 hash 另存于 `PAPER_VALIDATION_PACKAGE.json`，在 Git 和 Release 中各保留一份。
权重是训练状态而不是可信可执行源码；加载自己的归档时优先使用 `torch.load(..., weights_only=True)`。
复现实验仍需按数据授权自行获得相同数据、安装依赖并取回记录的上游版本。
尤其 WaveMLP 上游未发现许可证，源码仅本地导入，未复制到本仓库。

## 发布门槛

[publish_validation.py](publish_validation.py) 在注册矩阵完整、无审计错误且保存权重检查完成后才打包。
上传前再次校验报告、清单和附件；GitHub 返回的附件长度及 SHA256 均核验。
已有 Release 或附件只能复用完全相同版本，脚本不覆盖或删除它们。

```bash
env PYTHONPATH=lib python 06-paper-validation/06-4-summary.py --require-complete --check-weights
env PYTHONPATH=lib python 06-paper-validation/06-7-export.py --require-complete
python artifacts/publish_validation.py package
# 提交报告与清单并推送后，再上传到该提交对应的 Release。
python artifacts/publish_validation.py upload
```

报告和配置可能包含执行时的本地路径作为来源证据，但归档解包路径均为仓库相对路径，不包含数据内容或密钥。
本轮结论不支持跨 LoRa/WiFi 通用方法主张；Shen confirm RX 已反复查看，不是新盲测。
