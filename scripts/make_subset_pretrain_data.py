import datasets
import glob
import os
from tqdm import tqdm

path = '/mnt/nas1/alsc_supply_tech_SlimPajama-627B_20240926201127'
save_path = '/home/admin/workspace/aop_lab/app_source/duchuheng/datasets/slimpajamas_subset'
split = 'train'
data_files = [ 
    file for file in glob.iglob(os.path.join(path, split, '**'), recursive=True)
    if '.' in os.path.basename(file)
]
data = datasets.load_dataset(path, data_files=data_files, streaming=True)['train'].shuffle(42)
data = datasets.Dataset.from_list([item for item in tqdm(data.take(10_0000))])
data.save_to_disk(save_path)
