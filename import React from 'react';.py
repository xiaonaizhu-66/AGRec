import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel
import pandas as pd
from tqdm import tqdm
import os

# --- 1. 配置区 ---
MODEL_PATH = './e5-large-v2'  # 指向你图片中的模型文件夹
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32  # 如果显存溢出(OOM)，调小这个值

# 待处理的数据集列表
DATASETS = {
    "steam": "steam_meta.csv",
    "baby": "Baby_Products_meta.csv",
    "musical": "Musical_Instruments_meta.csv",
    "yelp": "yelp_meta.csv"
}


# --- 2. 工具函数 (来源于你的 handler.py) ---
def average_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]


def get_embeddings(texts, tokenizer, model):
    # E5 模型要求在查询或非对称任务前加 "passage: " 前缀
    processed_texts = [f"passage: {t}" for t in texts]

    inputs = tokenizer(processed_texts, max_length=512, padding=True, truncation=True, return_tensors='pt').to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        embeddings = average_pool(outputs.last_hidden_state, inputs['attention_mask'])
        embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu()


# --- 3. 主程序 ---
def main():
    print(f"正在加载模型: {MODEL_PATH} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE)
    model.eval()

    for name, csv_file in DATASETS.items():
        if not os.path.exists(csv_file):
            print(f"⚠️ 跳过 {name}: 找不到文件 {csv_file}")
            continue

        print(f"\n🚀 正在处理数据集: {name}")
        df = pd.read_csv(csv_file)

        # 【重要】拼接文本。假设第一列是 ID，后面列是标题/描述
        # 如果你的 CSV 列名固定，可以改写成 df['title'] + " " + df['description']
        # 这里默认拼接除第一列(ID)外的所有列
        texts = df.iloc[:, 1:].fillna("").apply(lambda x: " ".join(x.astype(str)), axis=1).tolist()

        all_embeddings = []

        # 分批处理防止 OOM
        for i in tqdm(range(0, len(texts), BATCH_SIZE)):
            batch_texts = texts[i: i + BATCH_SIZE]
            batch_emb = get_embeddings(batch_texts, tokenizer, model)
            all_embeddings.append(batch_emb)

        # 合并并保存
        final_tensor = torch.cat(all_embeddings, dim=0)
        save_path = f"{name}_e5_embeddings.pt"
        torch.save(final_tensor, save_path)

        print(f"✅ {name} 处理完成！向量形状: {final_tensor.shape} -> 保存至: {save_path}")


if __name__ == "__main__":
    main()