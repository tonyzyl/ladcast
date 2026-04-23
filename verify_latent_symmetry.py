"""
验证潜空间的经度平移等变性是否成立。

如果 encoder(shift(x)) ≈ shift(encoder(x))，则对称性假设成立。
如果二者差异很大，则说明 circular shift 不是潜空间的真对称性，
SymDiff 的对称化反而引入了噪声。

使用方法：用你的 DC-AE encoder 替换下面的伪代码部分。
"""
import torch
import numpy as np

def verify_latent_equivariance(encoder, sample_batch, shift_amounts=[1, 5, 10, 15]):
    """
    encoder: DC-AE 的编码器，输入 [B, C, H, W] 输出 [B, C_lat, H_lat, W_lat]
    sample_batch: 一批物理场数据 [B, C, 121, 240]（或对应分辨率）
    """
    results = {}
    
    with torch.no_grad():
        z_original = encoder(sample_batch)  # [B, C_lat, H_lat, W_lat]
        
        for k in shift_amounts:
            # 方案A：先平移物理场，再编码
            x_shifted = torch.roll(sample_batch, shifts=k * 8, dims=-1)  
            # k*8 因为 240/30=8，物理场每8格对应潜空间1格
            z_from_shifted_x = encoder(x_shifted)
            
            # 方案B：先编码，再平移潜空间
            z_shifted = torch.roll(z_original, shifts=k, dims=-1)
            
            # 计算差异
            diff = (z_from_shifted_x - z_shifted).pow(2).mean().item()
            baseline = z_original.pow(2).mean().item()
            relative_error = diff / (baseline + 1e-8)
            
            results[k] = {
                'absolute_mse': diff,
                'relative_error': relative_error,
                'baseline_energy': baseline
            }
            print(f"Shift={k:2d} | MSE(encode∘shift, shift∘encode) = {diff:.6f} | "
                  f"relative = {relative_error:.4f} ({relative_error*100:.2f}%)")
    
    avg_rel = np.mean([v['relative_error'] for v in results.values()])
    print(f"\n平均相对误差: {avg_rel*100:.2f}%")
    if avg_rel > 0.05:
        print("⚠️  等变性不成立（>5%误差），经度平移不是潜空间的真对称性")
        print("   → SymDiff 的 circular shift 对称化在伤害模型")
    elif avg_rel > 0.01:
        print("⚠  等变性近似成立（1-5%误差），可能需要软对称化")
    else:
        print("✓  等变性良好成立（<1%误差），SymDiff 框架适用")
    
    return results


# ---- 如果你没有现成encoder可用，下面用随机数据模拟来展示逻辑 ----
if __name__ == "__main__":
    print("=" * 60)
    print("模拟测试（随机encoder，仅展示验证逻辑）")
    print("=" * 60)
    
    # 模拟：一个不等变的 encoder
    class FakeEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(4, 84, kernel_size=8, stride=8, padding=0)
        def forward(self, x):
            # 简单卷积不保证循环平移等变
            return self.conv(x)
    
    encoder = FakeEncoder()
    sample = torch.randn(2, 4, 120, 240)  # 模拟物理场
    
    verify_latent_equivariance(encoder, sample, shift_amounts=[1, 5, 10, 15])
