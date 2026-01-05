import warnings
warnings.filterwarnings('ignore')
from recbole.quick_start import run_recbole

print(">>> 正在启动 SASRec-Category ...")

# 手动配置参数
config_dict = {
    # ================= 1. 数据集配置 =================
    'data_path': './',             
    'dataset': 'amazon_baby_inter', 
    'field_separator': '\t',       
    
    # 同时加载 交互数据(inter) 和 商品属性数据(item)
    'load_col': {                  
        'inter': ['user_id', 'item_id', 'timestamp'],
        'item': ['item_id', 'category'] 
    },
    
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp',

    # ================= 2. 训练配置 =================
    'epochs': 50,                  
    'train_batch_size': 2048,      
    'learner': 'adam',             
    'learning_rate': 0.001,        
    'eval_batch_size': 512,
    #核心配置 RecBole 进行 Embedding 并拼接到 Item ID 上
    'selected_features': ['category'],
    
    # 融合方式：拼接 (concat) 
    'pooling_mode': 'concat',

    # ================= 3. 模型配置 =================
    'model': 'SASRec-category',             
    
    'n_layers': 2,                 
    'n_heads': 2,                  
    'hidden_size': 64,             
    'inner_size': 256,             

    # 强制关闭负采样
    'train_neg_sample_args': None, 
    
    # ================= 4. 评测配置 =================
    'eval_args': {
        'split': {'RS': [0.8, 0.1, 0.1]}, 
        'group_by': 'user',
        'order': 'TO',             
        'mode': 'full'             
    },
    'metrics': ['Recall', 'NDCG'], 
    'topk': [10],   
    'valid_metric': 'NDCG@10'                
}

# 运行
# RecBole 会自动在 log 文件夹下生成一个新的日志文件
run_recbole(model='SASRec', config_dict=config_dict)