import warnings
warnings.filterwarnings('ignore')
from recbole.quick_start import run_recbole

# ================= 配置 =================
config_dict = {
    'data_path': './',             
    'dataset': 'amazon_baby_inter', 
    'field_separator': '\t',       
    'load_col': {'inter': ['user_id', 'item_id', 'timestamp']},
    'USER_ID_FIELD': 'user_id',
    'ITEM_ID_FIELD': 'item_id',
    'TIME_FIELD': 'timestamp',
    'epochs': 20,                  
    'train_batch_size': 2048,      
    'learner': 'adam',             
    'learning_rate': 0.001,
    
    # BPR 需要负采样，使用默认配置即可，不要设为 None
    
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

# ================= 只跑 BPR =================
print("正在运行 BPR (矩阵分解)...")
print("这个需要几分钟，请耐心等待...")
run_recbole(model='BPR', config_dict=config_dict)