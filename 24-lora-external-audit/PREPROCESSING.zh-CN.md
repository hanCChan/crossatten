# 前处理定义与信息预算

共同原包：Fs=1MHz、BW=125kHz、SF7，N=8192。HDF5每行前8192项为I，后8192项为Q；先与历史缓存一致转float32，再构成复数。原标签31..40显式减31→0..9，不使用“最后10类”推断。包身份为`{rx}_train.h5:{raw_row}`。

每RX重新初始化 `default_rng(20261012)`，按原始TX31..40逐类无放回抽200行，再全局升序排序；每TX按行号升序前160训练、后40源验证。raw不连续的样本不能重新取前200或先随机80/20。两折各5台source；开发RX在另一个fold担任source是原固定协议，数据/权重/选模必须逐折隔离。

令 z[n]=I[n]+jQ[n]。逐包功率归一化为 z/sqrt(mean|z|²)；ASK loader使用max(power,1e−12)，其ASK末端另外采用单位总能量归一化，两者不能混同。

**共同Strict-IB投影**：

`P_IB(z)=IFFT(FFT(z)[k]·1{|fftfreq(N,1/Fs)[k]|≤62500})`。

这里保留DC和±512bin，共1025bin，输出长度仍8192；没有降采样或裁剪时间。输出舍入float32产生极小带外残差，测试记录能量比例和与既有torch投影的差值，不宣称浮点精确零。
Native表让各方法保留自己输入前处理；Strict-IB表先施加相同P_IB，再运行各自完整方法。原HStem自己已包含TrueIB，因此两表复用同一冻结H0结果，无需训练两遍。

这限制的是可获得的原始信息。ASK的全矩阵混合、非线性和窗函数之后可能产生带外分量，它们是带内输入的函数，不是从原始带外偷取的信息；不在每个模块后额外投影。没有把Strict-IB输出表称为所有层始终带限。

**Shen TMC CIS**：RMS归一化后，用128点boxcar窗、hop64，无边界填充、双边STFT，按SciPy spectrum尺度除128，fftshift后加1e−12。记S[k,t]为该谱，输出

`C[k,t]=log10(|S[k,t+1]/S[k,t]|²)`。

8192点产生127帧，商后126列；频率crop为`floor(128*.3):ceil(128*.7)=38:90`，52行，bin中心为−203125至195312.5Hz。窗泄漏意味着它不是理想砖墙带通。
本次通过AST仅提取原class并原样执行，没有运行其数据loader或main；与既有torch float64移植分别在native/strict输入比较。STFT的epsilon、尺度与padding都不能任意改变。

**TIFS2022 CIS**：窗256/hop128，原谱不加epsilon；102行来自`round(256*.3):round(256*.7)=77:179`，商后62列，输出102×62×1。没有把它强行改成52×126以套用TMC网络。原算法遇零分母可能非有限；本次实包和随机探针有限并不证明所有输入有限。若增加稳定项，需记录并独立命名。

**TIFS2023 CIS**：窗64/hop32，无crop，商后254列，转置为序列254×64。只核验当前SF7输入兼容性，不把这项检查等同于多SF完整训练复现。

**ASK LoRa频谱算子**：8192点双边FFT/fftshift；候选k的mask保留中心`c−k+1:c+k`，共2k−1bin。训练用Gumbel-softmax硬选择，推断用最大logit的候选；最大k4096，SKO抽取中心8191bin。

`U=QR(A).Q; K=U diag(exp(jθ)) Uᴴ; X_evolved=X_masked Kᵀ`。

散射回8192谱→IFFT→按sqrt(sum|z|²)+1e−8归一化。随后是作者torch CIS和ResNet；其STFT未除窗长，并在log中再加epsilon，不与TMC数值实现混用。LSG选择的窄频带不保证SKO输出仍只有那2k−1bin非零。

**IB-RFF适配边界**：不能把8192简单截成256，以免丢弃其余原始窗口；不能先换成HStem五通道来冒充FAN。需显式选择seq_len8192、SAF长度8192、patch及位置编码，并保留两阶段与原HSIC。未冻结这些配置时不输出其“完整官方LoRa方法”的分数。

以上定义的出处为 [UPSTREAM.json](UPSTREAM.json)中的原始文件和本仓库 `lib/xa/frontends.py`；数值形状与误差在 [FRONTEND_AUDIT.json](FRONTEND_AUDIT.json)。原始特征不拼接RX标识、CFO或SNR；RX标识仅可作为Shen对抗头的源监督。
