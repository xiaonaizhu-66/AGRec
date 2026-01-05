import gzip
import json
import pandas as pd

meta_path = r'C:\Users\15535\Downloads\meta_All_Beauty.jsonl.gz'
output_path = 'amazon_All_Beauty_inter.item'

print("正在读取 Meta 数据提取类目...")
data = []

with gzip.open(meta_path, 'rt', encoding='utf-8') as f:
    for line in f:
        try:
            js = json.loads(line)
            item_id = js.get('parent_asin', js.get('asin'))
            category = js.get('main_category', None)
            if item_id and category:
                data.append([item_id, category])
        except:
            continue

# 转 DataFrame 并去重、重置索引
df_item = pd.DataFrame(data, columns=['item_id', 'category'])
df_item = df_item.drop_duplicates(subset=['item_id']).reset_index(drop=True)

# 检查是否正常
print(df_item.head())
print(df_item['item_id'].sample(5))
