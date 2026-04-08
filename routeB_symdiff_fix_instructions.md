# RouteB SymDiff 诊断与修复指令

> **目标读者**：Claude Code（自动执行）
> **项目**：ladcast RouteB — 在 LaDCast 潜空间扩散模型上集成 SymDiff 随机对称化
> **当前状态**：SymDiff 集成后性能劣化，需要诊断根因并修复

---

## 0. 当前结果与核心矛盾

```
| model                     | train metric | valid latent MSE |
|---------------------------|-------------|------------------|
| non_symm_resnet           | 0.002463    | 0.004065         |  ← AR 监督最优
| routeB_diffusion_nonsymm  | 0.015208    | 0.009306         |  ← diffusion 次优
| routeB_symdiff            | 0.010958    | 0.021092         |  ← SymDiff 最差
```

**核心矛盾**：routeB_symdiff 的 train eps_mse（0.011）比 nonsymm（0.015）更低，说明模型在训练分布上学得更好。但 valid_latent_mse（0.021）反而比 nonsymm（0.009）高 2 倍多，说明采样/推理阶段出了严重问题。

**根因假设**：`torch.roll` 做经度循环平移不是 DC-AE 潜空间的真对称性。DC-AE 用 stride>1 的卷积压缩空间，`encoder(roll(x)) ≠ roll(encoder(x))`。训练时模型被迫拟合物理上不合理的 rolled 输入，推理时采样轨迹偏离合理分布。

---

## 1. 任务一：验证等变性假设（诊断）

### 1.1 要做什么

写一个脚本 `tools/verify_latent_equivariance.py`，验证 DC-AE 的编码器是否对经度循环平移等变。

### 1.2 逻辑

```
对于一批物理场样本 x（从 ERA5 或 RouteBLatentDataset 对应的原始数据取）:
  z = encoder(x)                           # 先编码
  for k in [1, 2, 5, 10, 15]:
    x_rolled = torch.roll(x, shifts=k*stride, dims=-1)   # 在物理场上 roll
    z_from_rolled = encoder(x_rolled)                      # 编码 rolled 物理场
    z_rolled = torch.roll(z, shifts=k, dims=-1)            # 在潜空间上 roll
    
    mse = MSE(z_from_rolled, z_rolled)
    baseline = z.pow(2).mean()
    relative_error = mse / baseline
    
    print(f"shift={k}, relative_error={relative_error:.4%}")

if 平均 relative_error > 5%:
    print("等变性不成立，circular roll 不是潜空间真对称性")
```

### 1.3 如果无法直接拿到 encoder

如果项目中没有方便加载 DC-AE encoder 的接口，可以用替代方案：直接在潜空间数据上做统计验证。

```python
# 替代方案：检查潜空间 roll 后的统计特性是否保持
# 如果 roll 是真对称性，那么 roll 后的数据应该和原始数据有相同的统计分布

from ladcast 的数据加载模块 import RouteBLatentDataset  # 按实际路径调整

dataset = RouteBLatentDataset(...)
x_in, x_out = dataset[0]  # 取一个样本

# 检查1：roll 后的 channel-wise 均值/方差是否变化
for k in [1, 5, 10, 15]:
    x_rolled = torch.roll(x_out, shifts=k, dims=-1)  # 在 W 维度 roll
    
    # 如果等变成立，每个空间位置的统计量应该只是平移了
    # 但全局均值应该不变
    orig_mean = x_out.mean(dim=(-2, -1))   # [C] 每通道空间均值
    rolled_mean = x_rolled.mean(dim=(-2, -1))
    mean_diff = (orig_mean - rolled_mean).abs().max().item()
    
    # 检查2：相邻位置的相关结构是否在 roll 后保持
    # 计算 W 方向的自相关
    orig_autocorr = (x_out[..., :-1] * x_out[..., 1:]).mean()
    rolled_autocorr = (x_rolled[..., :-1] * x_rolled[..., 1:]).mean()
    autocorr_diff = abs(orig_autocorr.item() - rolled_autocorr.item())
    
    print(f"shift={k}: mean_diff={mean_diff:.6f}, autocorr_diff={autocorr_diff:.6f}")
```

### 1.4 输出要求

脚本运行后打印清晰的结论：
```
=== Latent Equivariance Verification ===
shift=1:  relative_error=XX.XX%
shift=5:  relative_error=XX.XX%
shift=10: relative_error=XX.XX%
shift=15: relative_error=XX.XX%
Average relative error: XX.XX%
CONCLUSION: [PASS/FAIL] — circular roll [is/is not] a valid symmetry in latent space
```

---

## 2. 任务二：新增 data augmentation baseline

### 2.1 要做什么

新增一个 baseline `routeB_diffusion_aug`，它在训练时对 (cond, target) 做随机经度 roll 增强，但推理时不做任何变换。这能分离两个效应：
- **训练时的正则化效果**（data aug 提供）
- **推理时的对称化采样效果**（SymDiff 独有）

### 2.2 实现方式

在 `tools/train_routeB_symdiff.py` 中新增 `--symmetry_mode augmentation` 选项：

```python
# 训练时
if symmetry_mode == 'augmentation':
    # 对 cond 和 target 做相同的随机 roll
    k = torch.randint(0, W, (1,)).item()  # W = 潜空间经度维度大小
    cond = torch.roll(cond, shifts=k, dims=-1)
    x0 = torch.roll(x0, shifts=k, dims=-1)
    # 然后正常做 diffusion 训练（无 group_state，无对称化）
    
# 推理时
if symmetry_mode == 'augmentation':
    # 不做任何变换，正常采样
    # 和 identity 模式完全相同
```

**关键**：augmentation 模式下训练目标和 loss 与 `routeB_diffusion_nonsymm` 完全一致，唯一区别是输入经过了随机 roll。

### 2.3 在对比脚本中注册

在 `tools/compare_routeB_minimal_baselines.py` 中新增这个 baseline，使其出现在统一结果表中。

---

## 3. 任务三：实现频域相位旋转替代 circular roll

### 3.1 背景

即使 `torch.roll`（硬平移）不是潜空间的真对称性，频域相位旋转是一种"软对称化"操作，它：
- 连续可微，支持梯度反传
- 保持频率能量谱不变（只改变相位）
- 不依赖空间平移等变性假设
- 可以通过 gamma 网络学习最优相位偏移

### 3.2 新增文件：`ladcast/models/routeB_fourier_shift.py`

```python
"""
频域相位旋转：替代 torch.roll 的软对称化操作。

用法：
    shifter = FourierLongitudeShift()
    z_shifted = shifter.forward_shift(z, phase_angle)
    z_back = shifter.inverse_shift(z_shifted, phase_angle)
    # z_back ≈ z（浮点精度内相等）
"""

import torch
import torch.nn as nn
import torch.fft


class FourierLongitudeShift(nn.Module):
    """沿经度维度（最后一维）做频域相位旋转"""
    
    def forward_shift(self, z, phase_angle):
        """
        Args:
            z: [B, ...any intermediate dims..., W]  最后一维是经度
            phase_angle: [B] 每个样本的相位偏移，范围 [0, 2*pi)
        Returns:
            z_shifted: 同形状，相位旋转后的张量
        """
        W = z.shape[-1]
        
        # 沿经度做 rFFT
        Z = torch.fft.rfft(z, dim=-1)  # [..., W//2+1]
        
        # 构建相位旋转因子
        freq_indices = torch.arange(Z.shape[-1], device=z.device, dtype=z.dtype)
        
        # phase_angle: [B] -> [B, 1, 1, ..., 1, W//2+1]
        # 需要 broadcast 到 Z 的形状
        n_expand = z.ndim - 1  # 除了 batch 维之外的维数（不含最后的频率维）
        shape = [phase_angle.shape[0]] + [1] * (n_expand - 1) + [Z.shape[-1]]
        phase = phase_angle.view(shape) * freq_indices.view([1] * n_expand + [-1])
        
        rotation = torch.complex(torch.cos(phase), torch.sin(phase))
        
        # 应用旋转
        Z_shifted = Z * rotation
        
        # iRFFT 回空间域
        z_shifted = torch.fft.irfft(Z_shifted, n=W, dim=-1)
        
        return z_shifted
    
    def inverse_shift(self, z, phase_angle):
        """逆变换 = 负相位旋转"""
        return self.forward_shift(z, -phase_angle)


class LearnablePhaseGamma(nn.Module):
    """
    可学习的 gamma 网络，输出用于相位旋转的角度。
    
    满足 SymDiff 框架要求：
    - 使用递归对称化：gamma = sym_{gamma_0}(gamma_1)
    - gamma_0 = Haar 测度（均匀随机角度）
    - gamma_1 = 可学习网络（不需要等变）
    
    实际采样流程：
    1. 采样 phi_0 ~ Uniform(0, 2*pi)         # Haar 测度
    2. 用 phi_0 反变换输入: z' = shift(z, -phi_0)
    3. gamma_1 预测增量: delta = gamma_1(z', eta)
    4. 最终角度: phi = phi_0 + delta
    """
    
    def __init__(self, input_channels, hidden_dim=256, noise_dim=16):
        super().__init__()
        self.noise_dim = noise_dim
        
        # 对输入做全局池化（消除空间依赖，获得不变特征）
        self.encoder = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),  # 全局池化 H,W -> 1,1
            nn.Flatten(),
            nn.Linear(input_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )
        
        self.noise_proj = nn.Linear(noise_dim, hidden_dim)
        
        self.head = nn.Linear(hidden_dim, 1)
        
        # 关键：identity-biased 初始化
        # 初始时输出接近零 → gamma 接近不变换 → 训练稳定启动
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
    
    def forward(self, z, eta=None):
        """
        Args:
            z: [B, C, H, W] 潜空间张量（已经被 phi_0 反变换过）
            eta: [B, noise_dim] 外部噪声，None 时自动采样
        Returns:
            delta_angle: [B] 增量相位角
        """
        if eta is None:
            eta = torch.randn(z.shape[0], self.noise_dim, device=z.device)
        
        h = self.encoder(z)  # [B, hidden_dim]
        h = h + self.noise_proj(eta)
        
        raw = self.head(h).squeeze(-1)  # [B]
        
        # 用 tanh 限制输出范围在 [-pi, pi]
        delta_angle = torch.tanh(raw) * torch.pi
        
        return delta_angle
```

### 3.3 修改 `ladcast/models/routeB_symdiff_denoiser.py`

在现有 denoiser 中，找到所有使用 `torch.roll` 做经度平移的地方，替换为 `FourierLongitudeShift`。

**具体修改逻辑**：

找到当前代码中类似这样的模式：
```python
# 当前（硬 roll）
k = sample_random_shift(...)
z_rolled = torch.roll(z, shifts=k, dims=-1)      # 正变换
cond_rolled = torch.roll(cond, shifts=k, dims=-1)
pred = denoiser(z_rolled, cond_rolled, t)
pred_unrolled = torch.roll(pred, shifts=-k, dims=-1)  # 逆变换
```

替换为：
```python
# 新版（频域相位旋转）
fourier_shifter = FourierLongitudeShift()

# 递归对称化采样
phi_0 = torch.rand(B, device=z.device) * 2 * torch.pi  # Haar 测度
z_pre = fourier_shifter.forward_shift(z, -phi_0)        # phi_0 反变换

delta = gamma_net(z_pre, eta)                            # gamma_1 预测增量
phi = phi_0 + delta                                      # 最终角度

# 对称化去噪
z_shifted = fourier_shifter.forward_shift(z, -phi)
cond_shifted = fourier_shifter.forward_shift(cond, -phi)
pred = denoiser_backbone(z_shifted, cond_shifted, t)
pred_unshifted = fourier_shifter.forward_shift(pred, phi)  # 正变换回来
```

### 3.4 新增 symmetry_mode

在 `tools/train_routeB_symdiff.py` 的 `--symmetry_mode` 中新增选项：

```
已有：
  identity            → routeB_diffusion_nonsymm（不做对称化）
  stochastic          → routeB_symdiff（当前的硬 roll 版本）

新增：
  augmentation        → routeB_diffusion_aug（训练 roll 增强，推理不变换）
  fourier_stochastic  → routeB_symdiff_fourier（频域相位旋转版 SymDiff）
  fourier_haar        → routeB_symdiff_fourier_haar（gamma=Haar，不学习 gamma）
```

### 3.5 保留旧版 stochastic 模式

**不要删除**旧的 `stochastic`（硬 roll）模式。保留它用于对比。

---

## 4. 任务四：实现 gamma warmup 策略

### 4.1 背景

SymDiff 训练初期，如果 gamma 网络输出不稳定，会让 denoiser 看到质量很差的变换后输入，导致训练不稳定。

### 4.2 在训练脚本中实现

在 `tools/train_routeB_symdiff.py` 中：

```python
# 新增命令行参数
parser.add_argument('--gamma_warmup_steps', type=int, default=0,
                    help='前 N 步冻结 gamma 网络，只训练 denoiser')
parser.add_argument('--gamma_warmup_mode', type=str, default='haar',
                    choices=['identity', 'haar'],
                    help='warmup 期间的 gamma 行为：identity=不变换，haar=均匀随机')

# 在训练循环中
for step in range(total_steps):
    if step < args.gamma_warmup_steps:
        # Warmup 期间
        if args.gamma_warmup_mode == 'identity':
            phi = torch.zeros(B, device=device)
        elif args.gamma_warmup_mode == 'haar':
            phi = torch.rand(B, device=device) * 2 * torch.pi
        # 不计算 gamma 网络的梯度
        for p in gamma_net.parameters():
            p.requires_grad_(False)
    else:
        # 正常训练
        for p in gamma_net.parameters():
            p.requires_grad_(True)
        # 递归对称化采样 phi
        phi_0 = torch.rand(B, device=device) * 2 * torch.pi
        eta = torch.randn(B, gamma_net.noise_dim, device=device)
        z_pre = fourier_shifter.forward_shift(z_t, -phi_0)
        delta = gamma_net(z_pre, eta)
        phi = phi_0 + delta
    
    # 后续去噪和 loss 计算...
```

建议默认值：`--gamma_warmup_steps` 设为总步数的 20%，`--gamma_warmup_mode haar`。

---

## 5. 任务五：升级评估体系

### 5.1 新增 CRPS 评估

在评估代码中（`tools/compare_routeB_minimal_baselines.py` 或单独的评估模块），新增：

```python
def compute_crps(model, pipeline, cond, target, n_samples=20):
    """
    计算 Continuous Ranked Probability Score。
    
    CRPS = E|X - Y| - 0.5 * E|X - X'|
    其中 X, X' 是模型的独立采样，Y 是真值。
    
    CRPS 越低越好。对于确定性模型（AR），退化为 MAE。
    
    Args:
        model: denoiser 模型
        pipeline: 采样 pipeline（对 AR 模型传 None）
        cond: 条件输入 [B, C, H, W]
        target: 真值 [B, C, H, W]
        n_samples: 采样数量
    Returns:
        crps: 标量
    """
    samples = []
    for i in range(n_samples):
        if pipeline is not None:
            # Diffusion 模型：通过 pipeline 采样
            sample = pipeline.sample(model, cond)
        else:
            # AR 模型：确定性预测（只有一个输出）
            with torch.no_grad():
                sample = model(cond)
        samples.append(sample.detach())
    
    samples = torch.stack(samples, dim=0)  # [n_samples, B, C, H, W]
    
    # Term 1: E|X - Y|
    flat_dims = tuple(range(2, samples.ndim))  # 对 C, H, W 求均值
    abs_errors = (samples - target.unsqueeze(0)).abs().mean(dim=flat_dims)  # [n_samples, B]
    term1 = abs_errors.mean(dim=0)  # [B]
    
    # Term 2: E|X - X'|（使用无偏估计）
    n = samples.shape[0]
    pairwise_sum = torch.zeros(samples.shape[1], device=samples.device)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            diff = (samples[i] - samples[j]).abs().mean(dim=flat_dims[:-1])
            # flat_dims[:-1] 对应从 dim=1 开始的所有维度（C, H, W）
            pairwise_sum += (samples[i] - samples[j]).abs().mean(dim=tuple(range(1, samples.ndim - 1 + 1)))
            count += 1
    
    # 更简单的实现：
    pairwise_sum = torch.zeros(samples.shape[1], device=samples.device)
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            pairwise_sum += (samples[i] - samples[j]).abs().flatten(1).mean(1)
            count += 1
    term2 = pairwise_sum / max(count, 1)  # [B]
    
    crps = (term1 - 0.5 * term2).mean().item()
    return crps
```

### 5.2 新增 Spread-Skill 评估

```python
def compute_spread_skill(model, pipeline, cond, target, n_samples=20):
    """
    检查 ensemble spread 是否与 prediction skill 匹配。
    
    理想情况下 spread/skill ratio ≈ 1.0：
    - ratio >> 1: 模型过度不确定（过散）
    - ratio << 1: 模型过度自信（欠散）
    
    Returns:
        dict with keys: skill, spread, ratio
    """
    samples = []
    for i in range(n_samples):
        if pipeline is not None:
            sample = pipeline.sample(model, cond)
        else:
            with torch.no_grad():
                sample = model(cond)
        samples.append(sample.detach())
    
    samples = torch.stack(samples, dim=0)  # [n_samples, B, C, H, W]
    
    ensemble_mean = samples.mean(dim=0)  # [B, C, H, W]
    
    skill = (ensemble_mean - target).pow(2).flatten(1).mean(1)   # [B] MSE of ensemble mean
    spread = samples.var(dim=0).flatten(1).mean(1)               # [B] ensemble variance
    
    skill_val = skill.mean().item()
    spread_val = spread.mean().item()
    ratio = spread_val / (skill_val + 1e-10)
    
    return {
        'skill_mse': skill_val,
        'spread_var': spread_val,
        'spread_skill_ratio': ratio
    }
```

### 5.3 修改对比脚本的输出表

在 `tools/compare_routeB_minimal_baselines.py` 的输出表中新增列：

```
| model | train metric | valid MSE | valid CRPS | spread | skill | SS ratio |
```

对 AR 模型，CRPS 退化为 MAE，spread=0，SS ratio=0（标注 N/A）。

---

## 6. 任务六：新增 inference ablation 模式（针对频域版本）

### 6.1 在 pipeline 中新增推理模式

在 `ladcast/pipelines/pipeline_routeB_symdiff.py` 中，针对频域版本新增推理模式：

```python
# 已有模式（保留）：
#   identity         - 不做变换
#   random_single    - 单次随机变换
#   fixed_group      - 固定变换
#   group_mean_N     - N 次随机变换取均值

# 新增模式（频域版）：
#   fourier_identity       - 不做相位旋转
#   fourier_random_single  - 单次随机相位旋转
#   fourier_learned        - 用训练好的 gamma_net 预测相位（正式模式）
#   fourier_mean_N         - N 次随机相位旋转，采样后取均值
#   fourier_grid_N         - N 次均匀网格相位（0, 2π/N, 4π/N, ...），取均值
```

**`fourier_grid_N` 模式特别重要**——它等价于对整个经度环做均匀采样平均，理论上能最好地消除经度偏差。

### 6.2 实现 fourier_grid_N

```python
def sample_fourier_grid(model, gamma_net, fourier_shifter, pipeline, cond, n_grid=8):
    """
    在均匀相位网格上做 N 次去噪采样，取均值。
    
    相当于对称群上的数值积分（近似 Haar 平均）。
    """
    angles = torch.linspace(0, 2 * torch.pi * (1 - 1/n_grid), n_grid)  # 不含 2π
    
    B = cond.shape[0]
    accumulated = torch.zeros_like(cond)  # 用 cond 的形状初始化
    
    # 需要从一致的初始噪声开始
    noise_seed = torch.randn_like(cond)  # 固定初始噪声
    
    for phi in angles:
        phi_batch = phi.expand(B).to(cond.device)
        
        # 反变换条件输入
        cond_shifted = fourier_shifter.forward_shift(cond, -phi_batch)
        
        # 用相同的初始噪声采样（保证公平比较）
        sample = pipeline.sample_from_noise(
            model, cond_shifted, initial_noise=noise_seed
        )
        
        # 正变换回来
        sample_unshifted = fourier_shifter.forward_shift(sample, phi_batch)
        
        accumulated += sample_unshifted
    
    return accumulated / n_grid
```

---

## 7. 任务七：更新 smoke test

### 7.1 新增频域相位旋转的 smoke test

新建 `tools/test_routeB_fourier_shift_smoke.py`：

```python
"""
Smoke test：验证 FourierLongitudeShift 的正确性。

测试项：
1. forward_shift + inverse_shift = identity（往返一致性）
2. phase_angle=0 时输出等于输入
3. 梯度可以正常回传
4. LearnablePhaseGamma 的输出在合理范围内
5. identity-biased 初始化：初始输出接近零
"""

def test_roundtrip():
    """forward + inverse 应该还原输入"""
    shifter = FourierLongitudeShift()
    z = torch.randn(4, 84, 15, 30)
    phi = torch.rand(4) * 2 * torch.pi
    
    z_shifted = shifter.forward_shift(z, phi)
    z_back = shifter.inverse_shift(z_shifted, phi)
    
    error = (z - z_back).abs().max().item()
    assert error < 1e-5, f"Roundtrip error too large: {error}"
    print(f"✓ Roundtrip test passed (max error={error:.2e})")

def test_zero_phase():
    """phase=0 应该是恒等变换"""
    shifter = FourierLongitudeShift()
    z = torch.randn(4, 84, 15, 30)
    phi = torch.zeros(4)
    
    z_shifted = shifter.forward_shift(z, phi)
    error = (z - z_shifted).abs().max().item()
    assert error < 1e-6, f"Zero-phase error: {error}"
    print(f"✓ Zero-phase test passed (max error={error:.2e})")

def test_gradient_flow():
    """梯度应该能正常通过"""
    shifter = FourierLongitudeShift()
    z = torch.randn(4, 84, 15, 30, requires_grad=True)
    phi = torch.rand(4, requires_grad=True) * 2 * torch.pi
    
    z_shifted = shifter.forward_shift(z, phi)
    loss = z_shifted.sum()
    loss.backward()
    
    assert z.grad is not None, "No gradient for z"
    assert phi.grad is not None, "No gradient for phi"
    assert not torch.isnan(z.grad).any(), "NaN in z gradient"
    assert not torch.isnan(phi.grad).any(), "NaN in phi gradient"
    print("✓ Gradient flow test passed")

def test_gamma_identity_init():
    """初始化后 gamma 输出应该接近零"""
    gamma = LearnablePhaseGamma(input_channels=84)
    z = torch.randn(4, 84, 15, 30)
    
    with torch.no_grad():
        delta = gamma(z)
    
    max_angle = delta.abs().max().item()
    assert max_angle < 0.01, f"Initial gamma output too large: {max_angle}"
    print(f"✓ Identity-biased init test passed (max initial angle={max_angle:.4f})")

def test_energy_preservation():
    """相位旋转应该保持频谱能量"""
    shifter = FourierLongitudeShift()
    z = torch.randn(4, 84, 15, 30)
    phi = torch.rand(4) * 2 * torch.pi
    
    orig_energy = z.pow(2).sum(dim=-1)  # 每个位置在 W 上的能量
    shifted_energy = shifter.forward_shift(z, phi).pow(2).sum(dim=-1)
    
    # Parseval 定理：能量应该相等（在浮点精度内）
    rel_error = ((orig_energy - shifted_energy).abs() / (orig_energy.abs() + 1e-8)).max().item()
    assert rel_error < 0.01, f"Energy not preserved: relative error={rel_error}"
    print(f"✓ Energy preservation test passed (max relative error={rel_error:.4f})")

if __name__ == "__main__":
    test_roundtrip()
    test_zero_phase()
    test_gradient_flow()
    test_gamma_identity_init()
    test_energy_preservation()
    print("\n=== All smoke tests passed ===")
```

### 7.2 端到端 smoke test

新建 `tools/train_routeB_fourier_symdiff_smoke.py`：

用最小配置（2 个样本，5 步训练）验证频域 SymDiff 的完整训练+采样流程能跑通。结构参照现有的 `tools/train_routeB_symdiff_smoke.py`。

---

## 8. 任务八：更新文档和对比脚本

### 8.1 更新 `docs/routeB_minimal_experiments.md`

新增以下内容：

```markdown
## 扩展 Baseline 列表（v2）

| 编号 | model_name                    | symmetry_mode       | 说明                    |
|------|-------------------------------|--------------------|-----------------------|
| 1    | tiny_ar                       | N/A                | 最小 AR 基线            |
| 2    | non_symm_resnet               | N/A                | ResNet AR 基线          |
| 3    | routeB_diffusion_nonsymm      | identity           | 无对称性 diffusion       |
| 4    | routeB_diffusion_aug          | augmentation       | 训练增强 diffusion       |
| 5    | routeB_symdiff                | stochastic         | 硬 roll SymDiff（旧版）  |
| 6    | routeB_symdiff_fourier        | fourier_stochastic | 频域 SymDiff（新版）     |
| 7    | routeB_symdiff_fourier_haar   | fourier_haar       | 频域 Haar-only SymDiff  |

## 新增评估指标

- valid_latent_mse: 单步 MSE（已有）
- valid_crps: CRPS（新增，diffusion 核心指标）
- spread_var: ensemble 方差（新增）
- skill_mse: ensemble 均值 MSE（新增）
- spread_skill_ratio: 散度-技巧比（新增，理想值≈1.0）

## 新增对比结论

除了已有的 A、B 结论，新增：
- C: 频域 SymDiff 是否优于硬 roll SymDiff
- D: data augmentation 是否解释了大部分 SymDiff 收益
- E: 哪种 inference mode 在 CRPS 上最优
```

### 8.2 更新 `tools/compare_routeB_minimal_baselines.py`

新增结论逻辑：

```python
# 结论 C
fourier_mse = results['routeB_symdiff_fourier']['valid_latent_mse']
roll_mse = results['routeB_symdiff']['valid_latent_mse']
if fourier_mse < roll_mse:
    print(f"C: Fourier SymDiff IMPROVES over roll SymDiff; gap={roll_mse - fourier_mse:.6f}")
else:
    print(f"C: Fourier SymDiff does not improve over roll SymDiff; gap={fourier_mse - roll_mse:.6f}")

# 结论 D
aug_mse = results['routeB_diffusion_aug']['valid_latent_mse']
fourier_mse = results['routeB_symdiff_fourier']['valid_latent_mse']
nonsymm_mse = results['routeB_diffusion_nonsymm']['valid_latent_mse']
aug_gain = nonsymm_mse - aug_mse
symm_gain = nonsymm_mse - fourier_mse
if aug_gain > 0 and symm_gain > 0:
    aug_fraction = aug_gain / symm_gain
    print(f"D: Data augmentation explains {aug_fraction*100:.1f}% of SymDiff gain")
else:
    print(f"D: No clear gain decomposition (aug_gain={aug_gain:.6f}, symm_gain={symm_gain:.6f})")

# 结论 E（CRPS 最优推理模式）
crps_results = {}  # inference_mode -> crps value
for mode in inference_modes:
    crps_results[mode] = evaluate_crps_for_mode(...)
best_mode = min(crps_results, key=crps_results.get)
print(f"E: Best inference mode by CRPS: {best_mode} (CRPS={crps_results[best_mode]:.6f})")
```

---

## 9. 执行顺序

请按以下顺序执行：

```
Step 1: 实现 tools/verify_latent_equivariance.py          → 运行并报告结果
Step 2: 实现 ladcast/models/routeB_fourier_shift.py       → 包含 FourierLongitudeShift + LearnablePhaseGamma
Step 3: 实现 tools/test_routeB_fourier_shift_smoke.py     → 运行通过
Step 4: 修改 routeB_symdiff_denoiser.py                   → 新增频域模式
Step 5: 修改 tools/train_routeB_symdiff.py                → 新增 symmetry_mode 选项 + gamma warmup
Step 6: 修改 pipeline                                     → 新增频域推理模式
Step 7: 实现 tools/train_routeB_fourier_symdiff_smoke.py  → 端到端 smoke，运行通过
Step 8: 新增 augmentation baseline 逻辑
Step 9: 更新 compare 脚本和文档
Step 10: 更新 docs/routeB_minimal_experiments.md
```

每完成一个 Step，先运行对应的 smoke test 确认不 break 已有功能，再继续下一步。

---

## 10. 不要做的事情（负面清单）

1. **不要删除** 已有的 `stochastic`（硬 roll）模式。保留用于对比。
2. **不要修改** AR baseline（tiny_ar, non_symm_resnet）的任何代码。
3. **不要修改** `RouteBLatentDataset` 的数据加载逻辑。
4. **不要引入** rollout / 解码 / 多步序列。保持单步 1→1 的实验范围。
5. **不要修改** diffusion scheduler 的超参（noise schedule, sigma 等）。只改对称化相关的部分。
6. **不要在** 频域相位旋转中使用 `torch.roll` 作为 fallback。两者应该完全独立。
