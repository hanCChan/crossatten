# 运行环境诊断与可审查的准备步骤

2026-11-10核验：

- 原conda环境Python3.12/Torch2.5.0+cu124被Windows Smart App Control拦截，涉及asmjit.dll、torch_cpu.dll，错误4551，CodeIntegrity事件3033/3077。
- 项目已有Python3.11虚拟环境的依赖版本一致，其独立官方构建也被同一策略拦截，涉及c10.dll。没有替换、重命名或重签名这些DLL。
- 系统VerifiedAndReputablePolicyState=1；策略ID为0283ac0f-fff1-49ae-ada1-8a933130cad6，对应Microsoft的VerifiedAndReputableDesktop。
- NVIDIA Windows驱动可见RTX 4060 Laptop；驱动版本616.56。
- WSL未安装发行版；“虚拟机平台”组件状态为未启用，系统报告HypervisorPresent=True。安装WSL2涉及启用Windows系统组件，可能需要重启，尚未执行。

## 路线选择

继续原生Windows，需要先通过本机正常安全管理流程解决PyTorch受信任运行问题。Smart App Control没有逐应用例外开关；当前没有已验证的原生修复可以直接应用。官方仓库仍有相关签名兼容问题记录，重装或盲目升级不保证解决。

若选择受支持的Linux/WSL2运行方式，以下是准备好的步骤。它们尚未执行，也不会更改Windows应用控制策略。系统组件安装与重启需由用户明确选择并安排。

管理员PowerShell先安装WSL及Ubuntu发行版，按系统提示安排重启：

```powershell
wsl --install -d Ubuntu-24.04
```

首次启动Ubuntu时创建Linux用户名及密码。Ubuntu终端内准备基础组件：

```bash
sudo apt-get update
sudo apt-get install -y python3-venv
bash /mnt/d/crossatten/worktrees/s17-rx-distortion/17-rx-distortion/environment/setup_linux.sh
```

setup_linux.sh只在Linux内创建独立venv，安装固定Torch2.5.0/CUDA12.4与分析依赖，然后执行GPU合成前反向验证；不会启动科学训练。NVIDIA要求WSL使用Windows宿主GPU驱动，不在WSL内另外安装Linux显示驱动。

GPU检查通过后，还需要运行完整工程测试，并为Linux路径及运行环境单独登记。原S17 D0的STOP和统计门槛保持不变；后续相位干预需单独预注册。

来源：

- [Microsoft：Smart App Control常见问题](https://support.microsoft.com/en-us/windows/security/threat-malware-protection/smart-app-control-frequently-asked-questions)
- [Microsoft：内置应用控制策略](https://learn.microsoft.com/en-us/windows/security/application-security/application-control/app-control-for-business/operations/inbox-appcontrol-policies)
- [PyTorch：Windows签名兼容问题记录](https://github.com/pytorch/pytorch/issues/199711)
- [Microsoft：安装WSL](https://learn.microsoft.com/en-us/windows/wsl/install)
- [NVIDIA：CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/)
