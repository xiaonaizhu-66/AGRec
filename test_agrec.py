import os, torch, pandas as pd, numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# --- 1. 先配置路径和基础变量 ---
# 注意：确保你的路径最后有斜杠，或者使用 os.path.join
ROOT = "d:/C盘的桌面/磨合期数据" 
DEVICE = torch.device("cpu")  # 为避免 GPU 内存分配失败，评估时改为 CPU
DATASETS = ["baby", "musical", "steam", "yelp"]

TRAIN_FILES = {
    "baby": f"{ROOT}/Baby_Products_inters_train.csv",
    "musical": f"{ROOT}/Musical_Instruments_inters_train.csv",
    "steam": f"{ROOT}/steam_inters_train.csv",
    "yelp": f"{ROOT}/yelp_inters_train.csv"
}

TEST_FILES = {
    "baby": f"{ROOT}/Baby_Products_inters_test.csv",
    "musical": f"{ROOT}/Musical_Instruments_inters_test.csv",
    "steam": f"{ROOT}/steam_inters_test.csv",
    "yelp": f"{ROOT}/yelp_inters_test.csv"
}

META_FILES = {
    "baby": f"{ROOT}/Baby_Products_meta.csv",
    "musical": f"{ROOT}/Musical_Instruments_meta.csv",
    "steam": f"{ROOT}/steam_meta.csv",
    "yelp": f"{ROOT}/yelp_meta.csv"
}

# 这里定义 MODEL_PATHS
MODEL_PATHS = {ds: f"{ROOT}/step9_{ds}_agrec_pro.pth" for ds in DATASETS}

# --- 2. 定义辅助函数 ---
def get_mapping(meta_path):
    meta_df = pd.read_csv(meta_path)
    item2idx = {str(rid): i for i, rid in enumerate(meta_df.iloc[:, 0].astype(str))}
    return item2idx


def get_unified_mapping(train_path, meta_path):
    df = pd.read_csv(train_path)
    u_list = sorted(df.iloc[:, 0].astype(str).unique().tolist())
    user2idx = {u: i for i, u in enumerate(u_list)}
    item2idx = get_mapping(meta_path)
    return user2idx, item2idx


def compute_topk_indices(user_final, i_c_all, i_f_all, proj_a, W_i, beta, topk=20, chunk_size=20000):
    best_scores = None
    best_idxs = None
    total_items = i_c_all.shape[0]
    offset = 0

    for st in range(0, total_items, chunk_size):
        en = min(total_items, st + chunk_size)
        i_c_chunk = i_c_all[st:en]
        i_f_chunk = i_f_all[st:en]

        i_c_proj = proj_a(i_c_chunk)
        i_c_aligned = F.normalize(F.linear(i_c_proj, W_i), dim=-1)
        i_f_norm = F.normalize(i_f_chunk, dim=-1)
        chunk_emb = F.normalize(i_f_norm + beta * i_c_aligned, dim=-1)

        chunk_scores = torch.matmul(user_final, chunk_emb.T).squeeze(0)

        if best_scores is None:
            k_cur = min(topk, en - st)
            best_scores, best_idxs = torch.topk(chunk_scores, k_cur, largest=True, sorted=True)
            best_idxs = best_idxs + offset
        else:
            all_scores = torch.cat([best_scores, chunk_scores], dim=0)
            all_indices = torch.cat([best_idxs, torch.arange(offset, offset + (en - st), device=all_scores.device)], dim=0)
            best_scores, topk_pos = torch.topk(all_scores, min(topk, all_scores.size(0)), largest=True, sorted=True)
            best_idxs = all_indices[topk_pos]

        offset += (en - st)

    return best_idxs.cpu().numpy()


dataset = "steam"
s8a = torch.load(f"step8a_{dataset}_pretrained.pth", map_location="cpu")  # 改成你的实际文件名
s8b = torch.load(f"step8b_{dataset}_llama.pth", map_location="cpu")  # 改成你的实际文件名
ckpt = torch.load(f"step9_{dataset}_agrec.pth", map_location="cpu")

print(f"=== {dataset} DEBUG ===")
print(f"s8a type: {type(s8a)}, shape/keys: {s8a.shape if hasattr(s8a, 'shape') else list(s8a.keys())[:10]}")
print(f"s8b type: {type(s8b)}, shape/keys: {s8b.shape if hasattr(s8b, 'shape') else list(s8b.keys())[:10]}")
print(f"ckpt keys: {list(ckpt.keys())[:20]}")
print(f"s8a stats: min={s8a.min():.4f}, max={s8a.max():.4f}, std={s8a.std():.4f}" if hasattr(s8a, 'min') else "s8a is dict")
print(f"s8b stats: min={s8b.min():.4f}, max={s8b.max():.4f}, std={s8b.std():.4f}" if hasattr(s8b, 'min') else "s8b is dict")


# --- 3. 定义主评估函数 ---
def evaluate_dataset(ds_name):
    print(f"\n正在评估数据集: {ds_name}")
    
    # 将模型加载移到函数内部！！此时 ds_name 和 MODEL_PATHS 已经存在了
    if not os.path.exists(MODEL_PATHS[ds_name]):
        print(f"找不到模型文件: {MODEL_PATHS[ds_name]}")
        return 

    # 加载 Checkpoint
    ckpt = torch.load(
        MODEL_PATHS[ds_name], 
        map_location=DEVICE, 
        weights_only=False  # 加上这一行解决安全加载限制
    )

    # 因为这个函数后面会完整实现，所以这里直接返回，避免重复定义覆盖
    #（旧 stub 已移除，实际逻辑在下面）
    return


@torch.no_grad()
def evaluate_dataset(ds_name):
    print(f"\n正在评估数据集: {ds_name}")
    if not os.path.exists(MODEL_PATHS[ds_name]):
        print(f"⚠️ 跳过 {ds_name}: 未找到权重文件 {MODEL_PATHS[ds_name]}")
        return

    # A. 加载模型与权重
    ckpt = torch.load(MODEL_PATHS[ds_name], map_location=DEVICE, weights_only=False)

    s8a_path = ckpt["step8a_path"]
    s8b_path = ckpt["step8b_path"]
    if not os.path.exists(s8a_path):
        s8a_path = os.path.join(ROOT, os.path.basename(s8a_path))
    if not os.path.exists(s8b_path):
        s8b_path = os.path.join(ROOT, os.path.basename(s8b_path))

    if not os.path.exists(s8a_path) or not os.path.exists(s8b_path):
        raise FileNotFoundError(f"找不到 step8a/step8b 文件: {s8a_path} / {s8b_path}")

    print('加载 s8a, s8b ...')
    s8a = torch.load(s8a_path, map_location=DEVICE, weights_only=False)
    s8b = torch.load(s8b_path, map_location=DEVICE, weights_only=False)
    print('s8a, s8b 加载完成')
    
    # 提取特征矩阵 (Move to GPU)
    u_c_all = s8a["user_emb.weight"].float().to(DEVICE)
    i_c_all = (s8a["item_llm_emb.weight"].float() + s8a["item_cf_emb.weight"].float()).to(DEVICE)
    u_f_all = s8b["user_seq_emb"].float().to(DEVICE)
    i_f_all = s8b["item_emb.weight"].float().to(DEVICE)

    n_items = min(i_c_all.shape[0], i_f_all.shape[0])
    if n_items != i_c_all.shape[0] or n_items != i_f_all.shape[0]:
        print(f"item emb count mismatch: i_c_all {i_c_all.shape[0]}, i_f_all {i_f_all.shape[0]} -> 此处取 {n_items}")
        i_c_all = i_c_all[:n_items]
        i_f_all = i_f_all[:n_items]

    u_hist_counts = torch.tensor(ckpt["u_hist_counts"]).float().to(DEVICE)

    # 粗粒度投影头 (保持与 step9_v2 对齐)
    proj_a = nn.Sequential(nn.Linear(u_c_all.shape[1], 512), nn.ReLU(), nn.Linear(512, 256)).to(DEVICE)
    proj_a[0].weight.data = s8a["proj.0.weight"].to(DEVICE)
    proj_a[0].bias.data = s8a["proj.0.bias"].to(DEVICE)
    proj_a[2].weight.data = s8a["proj.2.weight"].to(DEVICE)
    proj_a[2].bias.data = s8a["proj.2.bias"].to(DEVICE)
    proj_a.eval()

    # 路由器参数
    beta = torch.sigmoid(ckpt["router.beta"]).item() * 0.15
    W_u = ckpt["router.proj_align_user.weight"].to(DEVICE)
    W_i = ckpt["router.proj_align_item.weight"].to(DEVICE)
    
    # B. 加载测试数据
    user2idx, item2idx = get_unified_mapping(TRAIN_FILES[ds_name], META_FILES[ds_name])
    test_df = pd.read_csv(TEST_FILES[ds_name])
    
    # 获取测试样本 (User ID -> target Item ID)
    test_users = test_df.iloc[:, 0].astype(str).map(user2idx).values
    test_targets = test_df.iloc[:, 1].astype(str).map(item2idx).values
    
    print('i_c_all', i_c_all.shape, 'i_f_all', i_f_all.shape)
    print('注意：使用分块Top-K策略避免单次矩阵过大分配')

    # D. 遍历用户进行排名
    metrics = {"R@5": [], "N@5": [], "R@10": [], "N@10": [], "MRR": []}
    
    max_limit = min(len(test_users), 1)  # 仅先评估单样本
    for idx in range(max_limit):
        u_idx = test_users[idx]
        target_i = test_targets[idx]

        if pd.isna(u_idx) or pd.isna(target_i):
            continue

        u_idx = int(u_idx)
        target_i = int(target_i)

        # 用户侧特征融合 (获取该用户的 gate_score)
        hc = proj_a(u_c_all[u_idx].unsqueeze(0))
        hf = u_f_all[u_idx].unsqueeze(0)
        h_len = u_hist_counts[u_idx].unsqueeze(0)
        
        # 模拟 RouterModule 的 forward_user
        h_c_n = F.normalize(F.linear(hc, W_u), dim=-1)
        h_f_n = F.normalize(hf, dim=-1)
        
        # 路由逻辑 (简化实现，直接从 ckpt 获取逻辑参数计算)
        feat = torch.cat([h_c_n, h_f_n, torch.log1p(h_len).unsqueeze(-1)], dim=-1)
        # 需手动模拟 gate 层 (Linear+LayerNorm+ReLU+Linear+Sigmoid)
        # 这里为了演示，我们假设 router 已加载并直接调用
        gate_w1, gate_b1 = ckpt["router.router.0.weight"].to(DEVICE), ckpt["router.router.0.bias"].to(DEVICE)
        gate_w2, gate_b2 = ckpt["router.router.3.weight"].to(DEVICE), ckpt["router.router.3.bias"].to(DEVICE)
        
        g = torch.relu(F.linear(feat, gate_w1, gate_b1))
        gate_score = torch.sigmoid(F.linear(g, gate_w2, gate_b2))
        
        res_weight = gate_score * 0.2
        user_final = F.normalize(h_f_n + res_weight * h_c_n, dim=-1)

        # 计算得分与排名（分块Top-K）
        indices = compute_topk_indices(user_final, i_c_all, i_f_all, proj_a, W_i, beta, topk=20, chunk_size=20000)
        
        # 计算指标
        rank = np.where(indices == target_i)[0]
        rank = rank[0] + 1 if len(rank) > 0 else 100
        
        metrics["R@5"].append(1 if rank <= 5 else 0)
        metrics["N@5"].append(1/np.log2(rank + 1) if rank <= 5 else 0)
        metrics["R@10"].append(1 if rank <= 10 else 0)
        metrics["N@10"].append(1/np.log2(rank + 1) if rank <= 10 else 0)
        metrics["MRR"].append(1/rank if rank <= 10 else 0)

    # E. 打印结果
    print(f"\n📊 {ds_name.upper()} 结果:")
    for m, vals in metrics.items():
        print(f"{m}: {np.mean(vals):.4f}")



# --- 4. 最后运行循环 ---
if __name__ == "__main__":
    for ds in DATASETS:
        evaluate_dataset(ds)