import os, torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ════════════════════════════════════════════════════════════
# 1. 基础配置与路径
# ════════════════════════════════════════════════════════════
ROOT = "/root/autodl-tmp"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATASETS = ["baby", "musical", "steam", "yelp"]
DS_DISPLAY = {"baby": "Baby_Products", "musical": "Musical_Instruments", "steam": "steam", "yelp": "yelp"}

DATA_CONFIG = {
    "baby": {"meta": f"{ROOT}/Baby_Products_meta.csv", "train": f"{ROOT}/Baby_Products_inters_train.csv", "test": f"{ROOT}/Baby_Products_inters_test.csv"},
    "musical": {"meta": f"{ROOT}/Musical_Instruments_meta.csv", "train": f"{ROOT}/Musical_Instruments_inters_train.csv", "test": f"{ROOT}/Musical_Instruments_inters_test.csv"},
    "steam": {"meta": f"{ROOT}/steam_meta.csv", "train": f"{ROOT}/steam_inters_train.csv", "test": f"{ROOT}/steam_inters_test.csv"},
    "yelp": {"meta": f"{ROOT}/yelp_meta.csv", "train": f"{ROOT}/yelp_inters_train.csv", "test": f"{ROOT}/yelp_inters_test.csv"},
}

# ════════════════════════════════════════════════════════════
# 2. 核心组件：Pro 版残差门控模型
# ════════════════════════════════════════════════════════════
class RouterModule(nn.Module):
    def __init__(self, dim_a, dim_b):
        super().__init__()
        self.proj_align_user = nn.Linear(dim_a, dim_b, bias=False)
        self.proj_align_item = nn.Linear(dim_a, dim_b, bias=False)
        # 维度对齐：2 * dim_b + 1 (Coarse + Fine + Log_History_Len)
        self.gate = nn.Sequential(
            nn.Linear(2 * dim_b + 1, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
        self.beta = nn.Parameter(torch.zeros(1))

    def load_from_state9(self, s9):
        """精准装填 Step 9 导出的权重"""
        self.proj_align_user.weight.data = s9["router.proj_align_user.weight"].float().to(DEVICE)
        self.proj_align_item.weight.data = s9["router.proj_align_item.weight"].float().to(DEVICE)
        self.gate[0].weight.data = s9["router.router.0.weight"].float().to(DEVICE)
        self.gate[0].bias.data = s9["router.router.0.bias"].float().to(DEVICE)
        self.gate[3].weight.data = s9["router.router.3.weight"].float().to(DEVICE)
        self.gate[3].bias.data = s9["router.router.3.bias"].float().to(DEVICE)
        self.beta.data = s9["router.beta"].float().to(DEVICE)

def build_proj_a(state_8a):
    """重建 Step 8a 的投影层"""
    proj = nn.Sequential(
        nn.Linear(state_8a["proj.0.weight"].shape[1], 512),
        nn.ReLU(),
        nn.Linear(512, 256),
    )
    proj[0].weight.data = state_8a["proj.0.weight"].float()
    proj[0].bias.data = state_8a["proj.0.bias"].float()
    proj[2].weight.data = state_8a["proj.2.weight"].float()
    proj[2].bias.data = state_8a["proj.2.bias"].float()
    return proj.eval().to(DEVICE)

# ════════════════════════════════════════════════════════════
# 3. 核心评估逻辑
# ════════════════════════════════════════════════════════════
@torch.no_grad()
def run_variant(variant, state9, state8a, state8b, loader, n_items_cfg, n_users_cfg, u_hist_counts):
    dim_a, dim_b = int(state9["dim_a"]), int(state9["dim_b"])
    n_users, n_items = state9.get("n_users", n_users_cfg), state9.get("n_items", n_items_cfg)
    proj_a = build_proj_a(state8a)

    # 1. 加载 Embedding (带安全边界检查)
    def safe_get_emb(tensor, target_len):
        curr_len = tensor.shape[0]
        if curr_len >= target_len: return tensor[:target_len]
        pad = torch.zeros(target_len - curr_len, tensor.shape[1])
        return torch.cat([tensor, pad], dim=0)

    u_emb_a = safe_get_emb(state8a["user_emb.weight"].float(), n_users).to(DEVICE)
    i_raw_a = (safe_get_emb(state8a["item_llm_emb.weight"].float(), n_items) + 
               safe_get_emb(state8a["item_cf_emb.weight"].float(), n_items)).to(DEVICE)
    u_emb_b = safe_get_emb(state8b["user_seq_emb"].float(), n_users).to(DEVICE)
    i_emb_b = safe_get_emb(state8b["item_emb.weight"].float(), n_items).to(DEVICE)

    if variant == "Coarse-Only":
        h_final, i_final = F.normalize(proj_a(u_emb_a), dim=-1), F.normalize(proj_a(i_raw_a), dim=-1)
    elif variant == "Fine-Only":
        h_final, i_final = F.normalize(u_emb_b, dim=-1), F.normalize(i_emb_b, dim=-1)
    else: # AGRec (Ours) - 智能残差融合
        rm = RouterModule(dim_a, dim_b).to(DEVICE).eval()
        rm.load_from_state9(state9)
        u_hist_tensor = torch.from_numpy(u_hist_counts).float().to(DEVICE)

        hc_n = F.normalize(rm.proj_align_user(proj_a(u_emb_a)), dim=-1)
        hf_n = F.normalize(u_emb_b, dim=-1)
        
        # 核心：根据历史长度自适应生成 gate_score
        feat = torch.cat([hc_n, hf_n, torch.log1p(u_hist_tensor).unsqueeze(-1)], dim=-1)
        gate_score = rm.gate(feat)
        res_weight = gate_score * 0.2  # 匹配 Pro 版训练比例
        h_final = F.normalize(hf_n + res_weight * hc_n, dim=-1)

        ic_n = F.normalize(rm.proj_align_item(proj_a(i_raw_a)), dim=-1)
        if_n = F.normalize(i_emb_b, dim=-1)
        b = torch.sigmoid(rm.beta) * 0.15
        i_final = F.normalize(if_n + b * ic_n, dim=-1)

    # 计算排名
    all_ranks = []
    for u_batch, i_pos in loader:
        u_batch = u_batch.to(DEVICE)
        i_pos = i_pos.to(DEVICE).clamp(0, n_items - 1)
        scores = h_final[u_batch] @ i_final.T
        tgt = scores[torch.arange(len(u_batch)), i_pos].unsqueeze(1)
        all_ranks.append((scores > tgt).sum(1).cpu())
    
    ranks = torch.cat(all_ranks)
    return {"Recall@10": (ranks < 10).float().mean().item(), 
            "NDCG@10": (1.0 / torch.log2(ranks.float() + 2))[ranks < 10].sum().item() / len(ranks)}

# ════════════════════════════════════════════════════════════
# 4. 主程序：加载与汇总
# ════════════════════════════════════════════════════════════
def main():
    results = []
    for ds in DATASETS:
        print(f"\n🚀 正在评估 Pro 版模型: {DS_DISPLAY[ds]}")
        pth9 = f"{ROOT}/step9_{ds}_agrec_pro.pth"
        if not os.path.exists(pth9): 
            print(f"  ⚠️ 跳过：找不到权重文件 {pth9}"); continue
        
        s9 = torch.load(pth9, map_location="cpu", weights_only=False)
        u_hist_counts = s9["u_hist_counts"] # 关键：读取历史长度数据
        s8a = torch.load(s9["step8a_path"], map_location="cpu")
        s8b = torch.load(s9["step8b_path"], map_location="cpu")

        # 准备测试集 Loader (严格对齐映射逻辑)
        train_df = pd.read_csv(DATA_CONFIG[ds]["train"])
        u_list = sorted(train_df.iloc[:,0].astype(str).unique().tolist())
        user2idx = {u: i for i, u in enumerate(u_list)}
        meta = pd.read_csv(DATA_CONFIG[ds]["meta"])
        item2idx = {str(iid): i for i, iid in enumerate(meta.iloc[:, 0].astype(str))}

        test_df = pd.read_csv(DATA_CONFIG[ds]["test"])
        test_df['u'] = test_df.iloc[:,0].astype(str).map(user2idx)
        test_df['i'] = test_df.iloc[:,1].astype(str).map(item2idx)
        test_df = test_df.dropna(subset=['u', 'i'])
        loader = DataLoader(list(zip(test_df['u'].astype(int), test_df['i'].astype(int))), batch_size=1024)

        row = {"Dataset": DS_DISPLAY[ds]}
        for var in ["Coarse-Only", "Fine-Only", "AGRec (Ours)"]:
            m = run_variant(var, s9, s8a, s8b, loader, len(item2idx), len(user2idx), u_hist_counts)
            row.update({f"{var}_{k}": v for k, v in m.items()})
            print(f"  {var:<15} | R@10: {m['Recall@10']:.4f} | N@10: {m['NDCG@10']:.4f}")
        results.append(row)

    # 最终汇总
    df = pd.DataFrame(results)
    print("\n" + "=" * 45 + "\nPRO 版最终均值汇总\n" + "=" * 45)
    for var in ["Coarse-Only", "Fine-Only", "AGRec (Ours)"]:
        print(f"{var:<15} | R@10: {df[f'{var}_Recall@10'].mean():.4f} | N@10: {df[f'{var}_NDCG@10'].mean():.4f}")

if __name__ == "__main__":
    main()
