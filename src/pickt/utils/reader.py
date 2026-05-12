# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-04-28

import os
import json
import pickle
import pandas as pd

from omegaconf import OmegaConf


def load_data(data_dir):
    _, ext = os.path.splitext(data_dir)
    ext = ext.lower()

    if ext == ".csv":
        data = pd.read_csv(data_dir)
    elif ext in (".pickle", ".pkl"):
        with open(data_dir, "rb") as f:
            data = pickle.load(f)
    elif ext == ".json":
        with open(data_dir, "r") as f:
            data = json.load(f)
    elif ext in (".xls", ".xlsx"):
        data = pd.read_excel(data_dir)
    elif ext == ".yaml":
        data = OmegaConf.load(data_dir)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    return data


def save_json(save_path, file_name, data, indent=4):
    with open(os.path.join(save_path, file_name), "w", encoding="utf-8") as output_file:
        json.dump(data, output_file, indent=indent, ensure_ascii=False)


def create_folder_if_not_exists(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)