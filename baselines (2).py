"""
传统基线 + L_orth 消融 (GPU)
==============================
Part A: SASRec, Bert4Rec, GRU4Rec (RecBole 框架)
Part B: L_orth 消融 (有/无正交损失的 Router)

预计: Part A ~3h, Part B ~40min, 总计 ~4h
先装 RecBole: pip install recbole
"""

import os, sys, time, json
import pandas as pd
import numpy as np

# ============================================================
# PART A: 传统基线 (RecBole)
# ============================================================

def install_recbole():
    os.system("pip install recbole -q")
    print("RecBole installed.")

def prepare_recbole_data(train_csv, val_csv, test_csv, dataset_name, output_dir="recbole_data"):
    """将 CSV 转换为 RecBole 格式 (.inter 文件)"""
    os.makedirs(f"{output_dir}/{dataset_name}", exist_ok=True)

    all_dfs = []
    for csv_path, label in [(train_csv, "train"), (val_csv, "val"), (test_csv, "test")]:
        df = pd.read_csv(csv_path)
        uc = [c for c in df.columns if "user" in c.lower()][0]
        ic = [c for c in df.columns if "item" in c.lower()][0]
        tc = [c for c in df.columns if "time" in c.lower()]
        
        renamed = df.rename(columns={uc: "user_id:token", ic: "item_id:token"})
        if tc:
            renamed = renamed.rename(columns={tc[0]: "timestamp:float"})
            renamed = renamed[["user_id:token", "item_id:token", "timestamp:float"]]
        else:
            renamed["timestamp:float"] = np.arange(len(renamed))
            renamed = renamed[["user_id:token", "item_id:token", "timestamp:float"]]
        
        renamed["_split"] = label
        all_dfs.append(renamed)

    full = pd.concat(all_dfs, ignore_index=True)
    
    # RecBole 需要一个 .inter 文件
    inter_path = f"{output_dir}/{dataset_name}/{dataset_name}.inter"
    full.drop(columns=["_split"]).to_csv(inter_path, sep="\t", index=False)
    
    # 保存 split indices
    train_n = len(all_dfs[0])
    val_n = len(all_dfs[1])
    test_n = len(all_dfs[2])
    
    print(f"  {dataset_name}: {train_n} train, {val_n} val, {test_n} test → {inter_path}")
    return output_dir, train_n, val_n, test_n


def run_recbole_model(model_name, dataset_name, output_dir, train_n, val_n, test_n):
    """用 RecBole 训练和评估一个模型"""
    try:
        from recbole.quick_start import run_recbole
        from recbole.config import Config
    except ImportError:
        print("  安装 RecBole...")
        install_recbole()
        from recbole.quick_start import run_recbole
        from recbole.config import Config

    config_dict = {
        "model": model_name,
        "dataset": dataset_name,
        "data_path": output_dir,
        
        # 数据设置
        "load_col": {"inter": ["user_id", "item_id", "timestamp"]},
        "USER_ID_FIELD": "user_id",
        "ITEM_ID_FIELD": "item_id",
        "TIME_FIELD": "timestamp",
        
        # 评估
        "eval_args": {
            "split": {"LS": "valid_and_test"},
            "group_by": "user",
            "order": "TO",  # temporal order
            "mode": "full",
        },
        "metrics": ["Recall", "NDCG", "MRR"],
        "topk": [5, 10, 20],
        "valid_metric": "NDCG@10",
        
        # 训练
        "epochs": 100,
        "train_batch_size": 2048,
        "eval_batch_size": 4096,
        "learning_rate": 0.001,
        "eval_step": 5,
        "stopping_step": 10,
        
        # 模型通用
        "embedding_size": 64,
        "train_neg_sample_args": {"distribution": "uniform", "sample_num": 1},
        
        # GPU
        "device": "cuda" if __import__("torch").cuda.is_available() else "cpu",
        
        # 减少输出
        "show_progress": True,
        "log_wandb": False,
    }

    # 模型特定参数
    if model_name == "SASRec":
        config_dict.update({
            "n_layers": 2,
            "n_heads": 2,
            "hidden_size": 64,
            "inner_size": 256,
            "hidden_dropout_prob": 0.5,
            "attn_dropout_prob": 0.5,
            "max_seq_length": 50,
            "loss_type": "BPR",
        })
    elif model_name == "BERT4Rec":
        config_dict.update({
            "n_layers": 2,
            "n_heads": 2,
            "hidden_size": 64,
            "inner_size": 256,
            "hidden_dropout_prob": 0.5,
            "attn_dropout_prob": 0.5,
            "max_seq_length": 50,
            "mask_ratio": 0.2,
            "loss_type": "CE",
        })
    elif model_name == "GRU4Rec":
        config_dict.update({
            "embedding_size": 64,
            "hidden_size": 128,
            "num_layers": 1,
            "dropout_prob": 0.3,
            "loss_type": "BPR",
        })

    print(f"\n  训练 {model_name} on {dataset_name}...")
    t0 = time.time()
    
    try:
        result = run_recbole(config_dict=config_dict)
        elapsed = time.time() - t0
        
        # 提取结果
        test_result = result["test_result"]
        metrics = {}
        for key, val in test_result.items():
            # RecBole 返回格式: recall@5, ndcg@10, mrr@10 等
            clean_key = str(key).lower()
            metrics[clean_key] = float(val)
        
        print(f"  {model_name} 完成 ({elapsed:.0f}s)")
        print(f"    R@10={metrics.get('recall@10', 0):.4f} N@10={metrics.get('ndcg@10', 0):.4f}")
        return metrics
        
    except Exception as e:
        print(f"  ❌ {model_name} 失败: {e}")
        return None


# ============================================================
# PART A 备选: 手写轻量实现 (如果 RecBole 装不上)
# ============================================================

def run_baselines_manual():
    """如果 RecBole 安装有问题, 用手写的轻量实现"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    class SeqDataset(Dataset):
        """序列推荐数据集: 为每个 user 构建交互序列"""
        def __init__(self, train_csv, user2idx, item2idx, max_len=50):
            df = pd.read_csv(train_csv)
            uc = [c for c in df.columns if "user" in c.lower()][0]
            ic = [c for c in df.columns if "item" in c.lower()][0]
            tc = [c for c in df.columns if "time" in c.lower()]
            
            if tc:
                df = df.sort_values(tc[0])
            
            self.seqs = {}
            for _, row in df.iterrows():
                u, i = str(row[uc]), str(row[ic])
                if u in user2idx and i in item2idx:
                    uid, iid = user2idx[u], item2idx[i]
                    self.seqs.setdefault(uid, []).append(iid)
            
            self.users = list(self.seqs.keys())
            self.max_len = max_len
            self.n_items = len(item2idx)
        
        def __len__(self): return len(self.users)
        
        def __getitem__(self, idx):
            uid = self.users[idx]
            seq = self.seqs[uid][-self.max_len-1:]
            target = seq[-1]
            input_seq = seq[:-1]
            
            # Pad
            pad_len = self.max_len - len(input_seq)
            input_seq = [0] * pad_len + input_seq
            
            return torch.tensor(input_seq, dtype=torch.long), torch.tensor(target, dtype=torch.long), uid

    class SASRecModel(nn.Module):
        def __init__(self, n_items, hidden=64, n_layers=2, n_heads=2, max_len=50, dropout=0.5):
            super().__init__()
            self.item_emb = nn.Embedding(n_items + 1, hidden, padding_idx=0)
            self.pos_emb = nn.Embedding(max_len, hidden)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads, 
                                                       dim_feedforward=256, dropout=dropout, batch_first=True)
            self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.dropout = nn.Dropout(dropout)
            self.norm = nn.LayerNorm(hidden)
            self.max_len = max_len
            self.n_items = n_items
        
        def forward(self, seq):
            # seq: [B, max_len]
            mask = (seq == 0)  # padding mask
            pos = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
            x = self.item_emb(seq) + self.pos_emb(pos)
            x = self.dropout(self.norm(x))
            
            # Causal mask
            sz = seq.shape[1]
            causal = torch.triu(torch.ones(sz, sz, device=seq.device), diagonal=1).bool()
            
            x = self.encoder(x, mask=causal, src_key_padding_mask=mask)
            return x[:, -1, :]  # Last position
        
        def predict(self, seq):
            h = self.forward(seq)  # [B, hidden]
            scores = h @ self.item_emb.weight[1:].T  # [B, n_items] (skip padding)
            return scores

    class GRU4RecModel(nn.Module):
        def __init__(self, n_items, hidden=128, emb_dim=64, n_layers=1, dropout=0.3):
            super().__init__()
            self.item_emb = nn.Embedding(n_items + 1, emb_dim, padding_idx=0)
            self.gru = nn.GRU(emb_dim, hidden, n_layers, batch_first=True, dropout=dropout if n_layers > 1 else 0)
            self.fc = nn.Linear(hidden, emb_dim)
            self.n_items = n_items
        
        def forward(self, seq):
            x = self.item_emb(seq)
            lengths = (seq != 0).sum(dim=1).clamp(min=1)
            output, _ = self.gru(x)
            # Get last valid hidden state
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, output.shape[2])
            h = output.gather(1, idx).squeeze(1)
            return self.fc(h)
        
        def predict(self, seq):
            h = self.forward(seq)
            return h @ self.item_emb.weight[1:].T

    class BERT4RecModel(nn.Module):
        """BERT4Rec: 双向 Transformer + Masked Item Prediction, 推理时预测末位"""
        def __init__(self, n_items, hidden=64, n_layers=2, n_heads=2, max_len=50, dropout=0.5, mask_ratio=0.2):
            super().__init__()
            self.n_items = n_items
            self.mask_token = n_items + 1          # 特殊 MASK token id
            self.item_emb = nn.Embedding(n_items + 2, hidden, padding_idx=0)  # +2: pad + mask
            self.pos_emb  = nn.Embedding(max_len, hidden)
            encoder_layer = nn.TransformerEncoderLayer(d_model=hidden, nhead=n_heads,
                                                       dim_feedforward=256, dropout=dropout, batch_first=True)
            self.encoder  = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
            self.dropout  = nn.Dropout(dropout)
            self.norm     = nn.LayerNorm(hidden)
            self.head     = nn.Linear(hidden, n_items)   # MLM head
            self.max_len  = max_len
            self.mask_ratio = mask_ratio

        def forward(self, seq):
            pad_mask = (seq == 0)
            pos = torch.arange(seq.shape[1], device=seq.device).unsqueeze(0)
            x = self.item_emb(seq) + self.pos_emb(pos)
            x = self.dropout(self.norm(x))
            x = self.encoder(x, src_key_padding_mask=pad_mask)
            return x   # [B, L, hidden]

        def forward_masked(self, seq):
            """训练: 随机 mask 部分 item，返回 (masked_seq, mask_positions, logits)"""
            B, L = seq.shape
            masked_seq = seq.clone()
            mask_pos   = torch.zeros_like(seq, dtype=torch.bool)
            for b in range(B):
                non_pad = (seq[b] != 0).nonzero(as_tuple=True)[0]
                if len(non_pad) == 0:
                    continue
                n_mask = max(1, int(len(non_pad) * self.mask_ratio))
                chosen = non_pad[torch.randperm(len(non_pad))[:n_mask]]
                masked_seq[b, chosen] = self.mask_token
                mask_pos[b, chosen]   = True
            h = self.forward(masked_seq)              # [B, L, hidden]
            logits = self.head(h)                     # [B, L, n_items]
            return logits, mask_pos

        def predict(self, seq):
            """推理: 把最后一个真实 item 替换为 MASK，预测该位置"""
            pred_seq = seq.clone()
            # 找每个样本最后一个非 pad 位置
            lengths = (seq != 0).sum(dim=1).clamp(min=1)
            for b in range(seq.shape[0]):
                pred_seq[b, lengths[b] - 1] = self.mask_token
            h = self.forward(pred_seq)                # [B, L, hidden]
            # 取最后一个真实位置的输出
            idx = (lengths - 1).unsqueeze(1).unsqueeze(2).expand(-1, 1, h.shape[2])
            h_last = h.gather(1, idx).squeeze(1)      # [B, hidden]
            return self.head(h_last)                   # [B, n_items]

    return SASRecModel, GRU4RecModel, BERT4RecModel


def train_and_eval_manual(ModelClass, model_name, train_csv, val_csv, test_csv,
                           user2idx, item2idx, n_items, device="cuda", epochs=50, max_len=50):
    """手写实现的训练+评估，支持 SASRec / GRU4Rec / BERT4Rec"""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    SASRecModel, GRU4RecModel, BERT4RecModel = run_baselines_manual()

    # ---- 构建序列 ----
    all_seqs = {}
    for csv_path in [train_csv, val_csv]:
        df = pd.read_csv(csv_path)
        uc = [c for c in df.columns if "user" in c.lower()][0]
        ic = [c for c in df.columns if "item" in c.lower()][0]
        tc = [c for c in df.columns if "time" in c.lower()]
        if tc:
            df = df.sort_values(tc[0])
        for _, row in df.iterrows():
            u, i = str(row[uc]), str(row[ic])
            if u in user2idx and i in item2idx:
                all_seqs.setdefault(user2idx[u], []).append(item2idx[i])

    # ---- Test data ----
    test_df = pd.read_csv(test_csv)
    uc = [c for c in test_df.columns if "user" in c.lower()][0]
    ic = [c for c in test_df.columns if "item" in c.lower()][0]
    test_users, test_items = [], []
    for _, row in test_df.iterrows():
        u, i = str(row[uc]), str(row[ic])
        if u in user2idx and i in item2idx:
            uid, iid = user2idx[u], item2idx[i]
            if uid in all_seqs and len(all_seqs[uid]) >= 2:
                test_users.append(uid)
                test_items.append(iid)

    # ---- 训练序列 ----
    train_seqs, train_targets = [], []
    for uid, seq in all_seqs.items():
        if len(seq) >= 2:
            input_seq = seq[:-1][-max_len:]
            pad = [0] * (max_len - len(input_seq)) + input_seq
            train_seqs.append(pad)
            train_targets.append(seq[-1])

    train_seqs    = torch.tensor(train_seqs,    dtype=torch.long)
    train_targets = torch.tensor(train_targets, dtype=torch.long)

    # ---- 实例化模型 ----
    if model_name == "SASRec":
        model = SASRecModel(n_items, max_len=max_len).to(device)
    elif model_name == "GRU4Rec":
        model = GRU4RecModel(n_items).to(device)
    elif model_name == "BERT4Rec":
        model = BERT4RecModel(n_items, max_len=max_len).to(device)
    else:
        return None

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    batch_size = 512 if model_name == "BERT4Rec" else 1024

    print(f"  训练 {model_name}... ({len(train_seqs)} sequences, device={device})")

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(train_seqs))
        total_loss = 0; nb = 0

        for s in range(0, len(train_seqs), batch_size):
            e   = min(s + batch_size, len(train_seqs))
            idx = perm[s:e]
            seq = train_seqs[idx].to(device)
            tgt = train_targets[idx].to(device)   # 1-indexed item ids

            if model_name == "BERT4Rec":
                # MLM loss: mask 随机位置，用 CE 预测原始 item
                logits, mask_pos = model.forward_masked(seq)  # [B,L,n_items], [B,L]
                # 对 mask 位置计算 CE（target 转 0-indexed）
                logits_flat = logits[mask_pos]          # [M, n_items]
                # 从原始 seq 取真实 item（1-indexed → 0-indexed）
                targets_flat = seq[mask_pos] - 1        # 注意：masked_seq 里该位置已是 mask_token
                # 用 train_seqs[idx] 取原始 item
                orig_seq = train_seqs[idx].to(device)
                targets_flat = orig_seq[mask_pos] - 1
                targets_flat = targets_flat.clamp(0, n_items - 1)
                loss = F.cross_entropy(logits_flat, targets_flat)
            else:
                scores = model.predict(seq)             # [B, n_items]
                pos_scores = scores[torch.arange(len(tgt)), tgt - 1]
                neg_idx    = torch.randint(0, n_items, (len(tgt),), device=device)
                neg_scores = scores[torch.arange(len(tgt)), neg_idx]
                loss = -F.logsigmoid(pos_scores - neg_scores).mean()

            opt.zero_grad(); loss.backward(); opt.step()
            total_loss += loss.item(); nb += 1

        if (ep + 1) % 10 == 0:
            print(f"    Ep {ep+1}/{epochs} loss={total_loss/nb:.4f}")

    # ---- 评估 ----
    model.eval()
    tu = torch.tensor(test_users, dtype=torch.long)
    ti = torch.tensor(test_items, dtype=torch.long)

    results = {"R@5": 0, "R@10": 0, "R@20": 0, "N@10": 0, "MRR": 0}
    vn = 0

    with torch.no_grad():
        for s in range(0, len(tu), 256):
            e    = min(s + 256, len(tu))
            uids = tu[s:e]
            gts  = ti[s:e]

            batch_seqs, valid_mask = [], []
            for uid in uids.tolist():
                if uid in all_seqs:
                    seq = all_seqs[uid][-max_len:]
                    pad = [0] * (max_len - len(seq)) + seq
                    batch_seqs.append(pad)
                    valid_mask.append(True)
                else:
                    batch_seqs.append([0] * max_len)
                    valid_mask.append(False)

            batch_seqs  = torch.tensor(batch_seqs, dtype=torch.long).to(device)
            scores      = model.predict(batch_seqs)     # [B, n_items]

            valid_mask  = torch.tensor(valid_mask)
            gts_valid   = gts[valid_mask].to(device)
            scores_valid = scores[valid_mask]
            if len(gts_valid) == 0: continue
            vn += len(gts_valid)

            gt_adj = (gts_valid - 1).clamp(0, scores_valid.shape[1] - 1)
            _, topk = torch.topk(scores_valid, 20, dim=1)

            for k in [5, 10, 20]:
                tk  = topk[:, :k]
                hit = (tk == gt_adj.unsqueeze(1)).any(1).float()
                pos = (tk == gt_adj.unsqueeze(1)).float().argmax(1)
                results[f"R@{k}"] += hit.sum().item()
                if k == 10:
                    results["N@10"] += (hit / torch.log2(pos.float() + 2)).sum().item()

            tk10 = topk[:, :10]
            hit  = (tk10 == gt_adj.unsqueeze(1)).any(1).float()
            pos  = (tk10 == gt_adj.unsqueeze(1)).float().argmax(1)
            results["MRR"] += (hit / (pos.float() + 1)).sum().item()

    for k in results: results[k] /= max(vn, 1)
    print(f"  {model_name}: R@5={results['R@5']:.4f} R@10={results['R@10']:.4f} "
          f"R@20={results['R@20']:.4f} N@10={results['N@10']:.4f} MRR={results['MRR']:.4f}")
    return results


# ============================================================
# PART B: L_orth 消融
# ============================================================

def run_lorth_ablation():
    """训练两个 Router: 有 L_orth (λ=0.1) 和 无 L_orth (λ=0)"""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    ALPHA_PRIOR = 0.1; LAMBDA_PRIOR = 0.5; BETA_INIT = 0.3
    EPOCHS = 15; BATCH = 2048; N_NEG = 20; LR = 3e-4; PATIENCE = 5
    TOP_K = [5, 10, 20]

    CONFIGS = {
        "steam": {"s8a":"step8a_steam_pretrained.pth","s8b":"step8b_steam_llama.pth",
                   "meta_csv":"steam_meta.csv","train_csv":"steam_inters_train.csv",
                   "val_csv":"steam_inters_val.csv","test_csv":"steam_inters_test.csv"},
        "musical": {"s8a":"step8a_musical_pretrained.pth","s8b":"step8b_musical_llama.pth",
                     "meta_csv":"Musical_Instruments_meta.csv","train_csv":"Musical_Instruments_inters_train.csv",
                     "val_csv":"Musical_Instruments_inters_val.csv","test_csv":"Musical_Instruments_inters_test.csv"},
        "yelp": {"s8a":"step8a_yelp_pretrained.pth","s8b":"step8b_yelp_llama.pth",
                  "meta_csv":"yelp_meta.csv","train_csv":"yelp_inters_train.csv",
                  "val_csv":"yelp_inters_val.csv","test_csv":"yelp_inters_test.csv"},
        "baby": {"s8a":"step8a_baby_pretrained.pth","s8b":"step8b_baby_llama.pth",
                  "meta_csv":"Baby_Products_meta.csv","train_csv":"Baby_Products_inters_train.csv",
                  "val_csv":"Baby_Products_inters_val.csv","test_csv":"Baby_Products_inters_test.csv"},
    }

    # 复用之前的工具函数
    def get_s8a_mapping(train_csv, meta_csv):
        df=pd.read_csv(train_csv); u_list=sorted([str(x) for x in df.iloc[:,0].dropna().unique()])
        user2idx={u:i for i,u in enumerate(u_list)}
        meta=pd.read_csv(meta_csv); raw=[str(x) for x in meta.iloc[:,0].dropna().tolist()]
        item2idx={r:i for i,r in enumerate(raw)}; return user2idx, item2idx

    def build_reindex(a,b,sz):
        inv={v:k for k,v in b.items()}; ri=torch.zeros(len(b),dtype=torch.long); m=0
        for ti in range(len(b)):
            o=inv.get(ti)
            if o and o in a and a[o]<sz: ri[ti]=a[o]; m+=1
        return ri,m

    def compute_emb(s8a,s8b,iri,uri,nub):
        w0,b0=s8a["proj.0.weight"],s8a["proj.0.bias"]; w2,b2=s8a["proj.2.weight"],s8a["proj.2.bias"]
        with torch.no_grad():
            x=s8a["item_llm_emb.weight"][iri]+s8a["item_cf_emb.weight"][iri]
            hci=F.normalize(F.relu(x@w0.T+b0)@w2.T+b2,dim=-1).to(DEVICE)
            ur=torch.zeros(nub,dtype=torch.long); ur[:len(uri)]=uri; ur=ur.clamp(max=s8a["user_emb.weight"].shape[0]-1)
            hcu=F.normalize(F.relu(s8a["user_emb.weight"][ur]@w0.T+b0)@w2.T+b2,dim=-1).to(DEVICE)
            hfi=F.normalize(s8b["item_emb.weight"],dim=-1).to(DEVICE)
            hfu=F.normalize(s8b["user_seq_emb"],dim=-1).to(DEVICE)
        return hci,hcu,hfi,hfu

    def load_split(csv,u2i,i2i,ni,nu):
        df=pd.read_csv(csv); uc=[c for c in df.columns if "user" in c.lower()][0]
        ic=[c for c in df.columns if "item" in c.lower()][0]; us,its=[],[]
        for _,r in df.iterrows():
            u,i=str(r[uc]),str(r[ic])
            if u in u2i and i in i2i:
                ui,ii=u2i[u],i2i[i]
                if ii<ni and ui<nu: us.append(ui); its.append(ii)
        return torch.tensor(us,dtype=torch.long),torch.tensor(its,dtype=torch.long)

    class BPR(Dataset):
        def __init__(self,csv,u2i,i2i,ni):
            df=pd.read_csv(csv); uc=[c for c in df.columns if "user" in c.lower()][0]
            ic=[c for c in df.columns if "item" in c.lower()][0]; self.pairs=[]; self.up={}
            for _,r in df.iterrows():
                u,i=str(r[uc]),str(r[ic])
                if u in u2i and i in i2i:
                    ui,ii=u2i[u],i2i[i]
                    if ii<ni: self.pairs.append((ui,ii)); self.up.setdefault(ui,set()).add(ii)
            self.n=ni
        def __len__(self): return len(self.pairs)
        def __getitem__(self,idx):
            u,p=self.pairs[idx]; neg=[]; ps=self.up.get(u,set())
            while len(neg)<N_NEG:
                n=np.random.randint(0,self.n)
                if n not in ps: neg.append(n)
            return u,p,neg

    class Router(nn.Module):
        def __init__(self,dc,df):
            super().__init__()
            self.align_user=nn.Linear(dc,df,bias=False); self.align_item=nn.Linear(dc,df,bias=False)
            self.mlp=nn.Sequential(nn.Linear(2*df,256),nn.ReLU(),nn.Dropout(0.1),nn.Linear(256,1))
            self.beta=nn.Parameter(torch.tensor(BETA_INIT))
            nn.init.constant_(self.mlp[-1].bias,-2.0)
        def get_alpha(self,hc,hf):
            hca=F.normalize(self.align_user(hc),dim=-1); hfn=F.normalize(hf,dim=-1)
            return torch.sigmoid(self.beta*self.mlp(torch.cat([hca,hfn],-1))).squeeze(-1),hca,hfn
        def fuse_items(self,hc,hf,am):
            return F.normalize(am*F.normalize(self.align_item(hc),dim=-1)+(1-am)*F.normalize(hf,dim=-1),dim=-1)

    def train_router(hci,hcu,hfi,hfu,dc,df,loader,vu,vi,ni,lambda_orth):
        torch.manual_seed(42); np.random.seed(42)
        router=Router(dc,df).to(DEVICE)
        opt=torch.optim.AdamW(router.parameters(),lr=LR,weight_decay=1e-4)
        sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EPOCHS)
        best_r=-1; best_st=None; noimp=0

        for ep in range(EPOCHS):
            router.train()
            for ub,pb,nb_ in loader:
                ub=ub.to(DEVICE); pb=pb.to(DEVICE); nb_t=torch.stack(nb_,1).to(DEVICE)
                alpha,hca,hfn=router.get_alpha(hcu[ub],hfu[ub])
                a=alpha.unsqueeze(1); hu=F.normalize(a*hca+(1-a)*hfn,dim=-1)
                hcip=F.normalize(router.align_item(hci[pb]),dim=-1); hfip=F.normalize(hfi[pb],dim=-1)
                hip=F.normalize(a*hcip+(1-a)*hfip,dim=-1); ps=(hu*hip).sum(-1)
                hcin=F.normalize(router.align_item(hci[nb_t]),dim=-1); hfin=F.normalize(hfi[nb_t],dim=-1)
                hin=F.normalize(a.unsqueeze(1)*hcin+(1-a.unsqueeze(1))*hfin,dim=-1)
                ns=(hu.unsqueeze(1)*hin).sum(-1)
                bpr=-F.logsigmoid(ps.unsqueeze(1)-ns).mean()
                orth=(hca*hfn).sum(-1).abs().mean()
                prior=((alpha-ALPHA_PRIOR)**2).mean()
                loss=bpr + lambda_orth*orth + LAMBDA_PRIOR*prior
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(router.parameters(),1.0); opt.step()
            sched.step()

            if (ep+1)%2==0:
                router.eval(); n=min(len(vu),5000); idx=torch.randperm(len(vu))[:n]
                vu2,vi2=vu[idx].to(DEVICE),vi[idx].to(DEVICE); hits=0
                CHUNK=10000; uc_flag=ni>50000
                with torch.no_grad():
                    for s in range(0,n,1024):
                        e=min(s+1024,n); u=vu2[s:e]; gt=vi2[s:e]; v=gt<ni; u,gt=u[v],gt[v]
                        if len(gt)==0: continue
                        hu2,al2=router.fuse_items,0; al2_,hca2,hfn2=router.get_alpha(hcu[u],hfu[u])
                        a2=al2_.unsqueeze(1); hu2=F.normalize(a2*hca2+(1-a2)*hfn2,dim=-1); aa=al2_.mean().item()
                        if uc_flag:
                            B=hu2.shape[0]; ts2=torch.full((B,10),-1e9,device=DEVICE)
                            tii2=torch.zeros(B,10,dtype=torch.long,device=DEVICE)
                            for cs in range(0,ni,CHUNK):
                                ce=min(cs+CHUNK,ni); hi2=router.fuse_items(hci[cs:ce],hfi[cs:ce],aa)
                                sc2=hu2@hi2.T; c22=torch.cat([ts2,sc2],1)
                                ci2=torch.cat([tii2,torch.arange(cs,ce,device=DEVICE).expand(B,-1)],1)
                                _,sel=torch.topk(c22,10,1); ts2=c22.gather(1,sel); tii2=ci2.gather(1,sel)
                            topk2=tii2
                        else:
                            hi2=router.fuse_items(hci,hfi,aa); _,topk2=torch.topk(hu2@hi2.T,10,1)
                        hits+=(topk2==gt.unsqueeze(1)).any(1).sum().item()
                vr=hits/n
                if vr>best_r: best_r=vr; best_st={k:v.cpu().clone() for k,v in router.state_dict().items()}; noimp=0
                else: noimp+=1
                if noimp>=PATIENCE: break

        if best_st: router.load_state_dict({k:v.to(DEVICE) for k,v in best_st.items()})
        return router, best_r

    def eval_router_full(router,hci,hcu,hfi,hfu,tu,ti,ni):
        router.eval(); n=len(tu); mk=20; CHUNK=10000; uc_flag=ni>50000
        res={"R@5":0,"R@10":0,"R@20":0,"N@10":0,"MRR":0}; vn=0
        with torch.no_grad():
            for s in range(0,n,2048):
                e=min(s+2048,n); u=tu[s:e].to(DEVICE); gt=ti[s:e].to(DEVICE)
                v=gt<ni; u,gt=u[v],gt[v]
                if len(gt)==0:
                    continue
                vn+=len(gt)
                alpha,hca,hfn=router.get_alpha(hcu[u],hfu[u])
                a=alpha.unsqueeze(1); hu=F.normalize(a*hca+(1-a)*hfn,dim=-1); aa=alpha.mean().item()
                if uc_flag:
                    B=hu.shape[0]; ts=torch.full((B,mk),-1e9,device=DEVICE)
                    tii=torch.zeros(B,mk,dtype=torch.long,device=DEVICE)
                    for cs in range(0,ni,CHUNK):
                        ce=min(cs+CHUNK,ni); hi=router.fuse_items(hci[cs:ce],hfi[cs:ce],aa)
                        sc=hu@hi.T; c2=torch.cat([ts,sc],1)
                        ci=torch.cat([tii,torch.arange(cs,ce,device=DEVICE).expand(B,-1)],1)
                        _,sel=torch.topk(c2,mk,1); ts=c2.gather(1,sel); tii=ci.gather(1,sel)
                    topk=tii
                else: hi=router.fuse_items(hci,hfi,aa); _,topk=torch.topk(hu@hi.T,mk,1)
                for k in [5,10,20]:
                    tk=topk[:,:k]; hit=(tk==gt.unsqueeze(1)).any(1).float()
                    pos=(tk==gt.unsqueeze(1)).float().argmax(1)
                    res[f"R@{k}"]+=hit.sum().item()
                    if k==10: res["N@10"]+=(hit/torch.log2(pos.float()+2)).sum().item()
                tk10=topk[:,:10]; hit=(tk10==gt.unsqueeze(1)).any(1).float()
                pos=(tk10==gt.unsqueeze(1)).float().argmax(1); res["MRR"]+=(hit/(pos.float()+1)).sum().item()
        for k in res: res[k]/=max(vn,1)
        return res

    # Main ablation loop
    print(f"\n{'='*60}")
    print(f"  PART B: L_orth 消融实验")
    print(f"{'='*60}")

    ablation_results = {}
    for ds_name, cfg in CONFIGS.items():
        missing = [k for k in cfg if not os.path.exists(cfg[k])]
        if missing:
            print(f"  跳过 {ds_name}: 缺少文件")
            continue

        print(f"\n  --- {ds_name.upper()} ---")
        s8a=torch.load(cfg["s8a"],map_location="cpu"); s8b=torch.load(cfg["s8b"],map_location="cpu")
        nia=s8a["item_llm_emb.weight"].shape[0]; nib=s8b["item_emb.weight"].shape[0]
        nub=s8b["user_seq_emb"].shape[0]; i2ib=s8b["item2idx"]; u2ib=s8b["user2idx"]
        u2ia,i2ia=get_s8a_mapping(cfg["train_csv"],cfg["meta_csv"])
        iri,_=build_reindex(i2ia,i2ib,nia); uri,_=build_reindex(u2ia,u2ib,s8a["user_emb.weight"].shape[0])
        hci,hcu,hfi,hfu=compute_emb(s8a,s8b,iri,uri,nub)
        dc,df_=hci.shape[1],hfi.shape[1]; del s8a,s8b; torch.cuda.empty_cache()

        ds=BPR(cfg["train_csv"],u2ib,i2ib,nib)
        loader=DataLoader(ds,batch_size=BATCH,shuffle=True,num_workers=4,pin_memory=True,drop_last=True)
        vu,vi=load_split(cfg["val_csv"],u2ib,i2ib,nib,nub)
        tu,ti=load_split(cfg["test_csv"],u2ib,i2ib,nib,nub)

        ablation_results[ds_name] = {}
        for lorth_val, label in [(0.1, "with_orth"), (0.0, "without_orth")]:
            print(f"    λ_orth={lorth_val} ({label})...")
            t0=time.time()
            router, val_r = train_router(hci,hcu,hfi,hfu,dc,df_,loader,vu,vi,nib,lorth_val)
            res = eval_router_full(router,hci,hcu,hfi,hfu,tu,ti,nib)
            elapsed=time.time()-t0
            ablation_results[ds_name][label] = res
            print(f"    {label}: R@10={res['R@10']:.4f} N@10={res['N@10']:.4f} Val={val_r:.4f} ({elapsed:.0f}s)")
            del router; torch.cuda.empty_cache()

    return ablation_results


# ============================================================
# PART A: Run Traditional Baselines
# ============================================================

def run_traditional_baselines():
    """用手写实现跑传统基线"""
    import torch
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CONFIGS = {
        "steam": {"train":"steam_inters_train.csv","val":"steam_inters_val.csv","test":"steam_inters_test.csv"},
        "musical": {"train":"Musical_Instruments_inters_train.csv","val":"Musical_Instruments_inters_val.csv","test":"Musical_Instruments_inters_test.csv"},
        "yelp": {"train":"yelp_inters_train.csv","val":"yelp_inters_val.csv","test":"yelp_inters_test.csv"},
        "baby": {"train":"Baby_Products_inters_train.csv","val":"Baby_Products_inters_train.csv","test":"Baby_Products_inters_test.csv"},
    }

    print(f"\n{'='*60}")
    print(f"  PART A: 传统基线 (SASRec, GRU4Rec)")
    print(f"{'='*60}")

    # 强制使用手写实现，跳过 RecBole（依赖太多，兼容性差）
    use_recbole = False
    print("  使用手写实现 (跳过 RecBole)")

    baseline_results = {}
    
    for ds_name, cfg in CONFIGS.items():
        if not all(os.path.exists(cfg[k]) for k in cfg):
            print(f"  跳过 {ds_name}"); continue

        print(f"\n  --- {ds_name.upper()} ---")
        
        # 构建 user/item 映射
        df = pd.read_csv(cfg["train"])
        uc = [c for c in df.columns if "user" in c.lower()][0]
        ic = [c for c in df.columns if "item" in c.lower()][0]
        
        all_users = set()
        all_items = set()
        for csv_path in [cfg["train"], cfg["val"], cfg["test"]]:
            if os.path.exists(csv_path):
                d = pd.read_csv(csv_path)
                all_users.update(d[uc].astype(str).unique())
                all_items.update(d[ic].astype(str).unique())
        
        user2idx = {u: i+1 for i, u in enumerate(sorted(all_users))}  # 1-indexed (0=pad)
        item2idx = {i: j+1 for j, i in enumerate(sorted(all_items))}
        n_items = len(item2idx)
        
        baseline_results[ds_name] = {}
        
        if use_recbole:
            outdir, tn, vn, tsn = prepare_recbole_data(cfg["train"], cfg["val"], cfg["test"], ds_name)
            for model_name in ["SASRec", "BERT4Rec", "GRU4Rec"]:
                res = run_recbole_model(model_name, ds_name, outdir, tn, vn, tsn)
                if res: baseline_results[ds_name][model_name] = res
        else:
            for model_name in ["SASRec", "GRU4Rec", "BERT4Rec"]:
                res = train_and_eval_manual(
                    None, model_name, cfg["train"], cfg["val"], cfg["test"],
                    user2idx, item2idx, n_items, DEVICE, epochs=50
                )
                if res: baseline_results[ds_name][model_name] = res

    return baseline_results


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    import torch
    torch.backends.cudnn.benchmark = True
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    all_results = {"baselines": {}, "ablation": {}}
    
    # Part B first (faster, ~40min)
    print("\n" + "="*60)
    print("  开始 Part B: L_orth 消融")
    print("="*60)
    t0 = time.time()
    ablation = run_lorth_ablation()
    all_results["ablation"] = ablation
    print(f"\n  Part B 完成 ({(time.time()-t0)/60:.0f} min)")

    # Part A (slower, ~2-3h)
    print("\n" + "="*60)
    print("  开始 Part A: 传统基线")
    print("="*60)
    t0 = time.time()
    baselines = run_traditional_baselines()
    all_results["baselines"] = baselines
    print(f"\n  Part A 完成 ({(time.time()-t0)/60:.0f} min)")

    # 保存
    with open("baselines_ablation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # 汇总
    print(f"\n{'='*85}")
    print(f"  L_orth 消融结果")
    print(f"{'='*85}")
    print(f"  {'数据集':<10} {'λ_orth=0.1':<20} {'λ_orth=0.0':<20} {'差异':<10}")
    print(f"  {'-'*60}")
    for ds in ablation:
        wo = ablation[ds].get("with_orth", {}).get("R@10", 0)
        woo = ablation[ds].get("without_orth", {}).get("R@10", 0)
        diff = (wo - woo) / max(woo, 1e-6) * 100
        print(f"  {ds:<10} R@10={wo:.4f}        R@10={woo:.4f}        {diff:+.2f}%")

    print(f"\n{'='*85}")
    print(f"  传统基线结果")
    print(f"{'='*85}")
    for ds in baselines:
        print(f"\n  {ds.upper()}:")
        for model in baselines[ds]:
            r = baselines[ds][model]
            print(f"    {model:<12} R@10={r.get('R@10', r.get('recall@10', 0)):.4f} "
                  f"N@10={r.get('N@10', r.get('ndcg@10', 0)):.4f}")
