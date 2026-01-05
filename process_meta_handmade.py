import gzip
import json
import pandas as pd
import os

# ================= 路径配置 =================
inter_path = r'C:\Users\15535\Desktop\论文代码\datasets\Handmade\Handmade.inter'
# 2. 原始 Meta 文件
meta_path = r'C:\Users\15535\Downloads\meta_Handmade_Products.jsonl.gz'
# 3. 输出位置 
output_path = r'C:\Users\15535\Desktop\论文代码\datasets\Handmade\Handmade.item'

print(f"1. 读取交互文件名单: {inter_path}")
if not os.path.exists(inter_path):
    print(" 找不到交互文件！请检查路径。")
    exit()

df_inter = pd.read_csv(inter_path, sep='\t', dtype=str)
valid_items = set(df_inter['item_id:token'].unique())
print(f"   需要寻找类目的商品数: {len(valid_items)}")

print(f"2. 扫描 Meta 文件...")
data = []
matched = 0

try:
    with gzip.open(meta_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                js = json.loads(line)
                item_id = js.get('parent_asin', js.get('asin'))
                
                if item_id in valid_items:
                    categories = js.get('categories', [])
                    final_cat = None
                    if categories:
                        final_cat = categories[-1] 
                    
                    # 保证 ID 对齐
                    if not final_cat:
                        final_cat = "Unknown"
                        
                    data.append([item_id, final_cat])
                    matched += 1
            except:
                continue
except FileNotFoundError:
    print(" 找不到 Meta 压缩包")
    exit()

# ================= 保存 =================
# 去重，确保每个商品只有一个类目
df_item = pd.DataFrame(data, columns=['item_id:token', 'category:token'])
df_item = df_item.drop_duplicates(subset=['item_id:token'])

# 再次过滤，确保只保留 inter 里有的商品
df_item = df_item[df_item['item_id:token'].isin(valid_items)]

print(f"3. 匹配完成。覆盖率: {len(df_item)} / {len(valid_items)} = {len(df_item)/len(valid_items)*100:.2f}%")

df_item.to_csv(output_path, sep='\t', index=False)
print(f" Handmade.item 已更新！")