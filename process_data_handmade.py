import gzip
import json
import pandas as pd
import os

# ================= 配置 =================
input_path = r'C:\Users\15535\Downloads\Handmade_Products.jsonl.gz'
output_path = r'C:\Users\15535\Desktop\论文代码\datasets\Handmade\Handmade.inter'

# ================= 1. 读取数据 =================
print(f"1. 读取数据: {input_path}")
data = []
count = 0

try:
    with gzip.open(input_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                js = json.loads(line.strip())
                user_id = js.get('user_id')
                item_id = js.get('parent_asin', js.get('asin'))
                timestamp = js.get('timestamp')
                
                if user_id and item_id and timestamp:
                    data.append([user_id, item_id, timestamp])
                    count += 1
            except:
                continue
            if count % 100000 == 0: print(f"   已读取 {count} 行...", end='\r')

except FileNotFoundError:
    print(" 路径不对")
    exit()

df = pd.DataFrame(data, columns=['user_id:token', 'item_id:token', 'timestamp:float'])
print(f"\n   原始数据量: {len(df)}")

# ================= 2. 防止数据泄露 去重 =================
print("2. 执行去重...")
df = df.sort_values(by='timestamp:float')
df = df.drop_duplicates(subset=['user_id:token', 'item_id:token'], keep='first')
print(f"   去重后数据量: {len(df)}")

# ================= 3. 用户过滤 =================
MIN_USER_INTER = 5 
print(f"3. 执行用户过滤 (Min Interactions >= {MIN_USER_INTER})...")

# 统计每个用户的购买数量
user_counts = df['user_id:token'].value_counts()
# 找到合格的用户
valid_users = user_counts[user_counts >= MIN_USER_INTER].index

# 过滤
df_clean = df[df['user_id:token'].isin(valid_users)]

# ================= 4. 最终检查 =================
print("-" * 30)
print(f"最终有效数据: {len(df_clean)}")
print(f"用户数: {df_clean['user_id:token'].nunique()}")
print(f"商品数: {df_clean['item_id:token'].nunique()}")

if len(df_clean) == 0:
    print(" 还是 0")
else:
    # 计算稀疏度
    sparsity = 1 - len(df_clean) / (df_clean['user_id:token'].nunique() * df_clean['item_id:token'].nunique())
    print(f"稀疏度: {sparsity:.4f}")
    print("-" * 30)
    
    # 自动创建目录并保存
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_clean.to_csv(output_path, sep='\t', index=False)
    print(f" 成功！文件已保存至: {output_path}")
