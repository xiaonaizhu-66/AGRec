import gzip
import json
import pandas as pd
import os

# ================= 配置区域 =================
inter_path = 'amazon_baby_inter.csv'
meta_path = r'C:\Users\15535\Downloads\meta_baby_Products.jsonl.gz'
output_path = 'amazon_baby_inter.item'


print(f"1. 读取交互文件名单: {inter_path}")

if not os.path.exists(inter_path):
    print(" 找不到交互文件")
    exit()

df_inter = pd.read_csv(inter_path, sep='\t', dtype=str)
valid_items = set(df_inter['item_id:token'].unique())
print(f"   需要寻找类目的商品数: {len(valid_items)}")

print(f"扫描 Meta 文件: {meta_path}")
data = []
matched = 0
scanned = 0

try:
    with gzip.open(meta_path, 'rt', encoding='utf-8') as f:
        for line in f:
            scanned += 1
            if scanned % 50000 == 0:
                print(f"   已扫描 {scanned} 行... (匹配: {matched})", end='\r')
                
            try:
                js = json.loads(line)
                item_id = js.get('parent_asin', js.get('asin'))
                
                if item_id in valid_items:
                    # 提取类目
                    categories = js.get('categories', [])
                    final_cat = None
                    
                    if categories and len(categories) > 0:
                        # 取最细的一级类目
                        cat_str = categories[-1]
                        if isinstance(cat_str, str):
                            final_cat = cat_str.strip()
                    
                    # 找到有效类目才保存
                    if final_cat:
                        data.append([item_id, final_cat])
                        matched += 1
                        
            except:
                continue
except FileNotFoundError:
    print(f" 找不到 Meta 文件: {meta_path}")
    exit()

print(f"\n3. 匹配完成。")

# ================= 质量检查 =================
if len(valid_items) > 0:
    coverage = matched / len(valid_items) * 100
else:
    coverage = 0

print("-" * 30)
print(f"覆盖率 (Coverage): {coverage:.2f}%")
print("-" * 30)

if coverage < 80:
    print(" 警告：覆盖率低！请检查 Review 和 Meta 文件是否来自同一个年份版本。")
else:
    print(" 数据质量优秀，可以开始训练")

# 保存
df_item = pd.DataFrame(data, columns=['item_id:token', 'category:token'])
df_item = df_item.drop_duplicates(subset=['item_id:token'])
df_item.to_csv(output_path, sep='\t', index=False)
print(f" Meta 文件已保存: {output_path}")