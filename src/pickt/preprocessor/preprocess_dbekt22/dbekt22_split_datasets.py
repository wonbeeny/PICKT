# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-05-13


import os
import time
import json
import random
import argparse
import polars as pl

from tqdm.auto import tqdm
from collections import defaultdict


# ratios=[7, 1.5, 1.5]
ratios=[8, 2, 0]

parser = argparse.ArgumentParser()

parser.add_argument('--data_args_path', default='', help='data_args 경로')
parser.add_argument('--preprocessed_data_path', default='', help='전처리되어 저장된 데이터의 경로')
parser.add_argument('--save_path', default='', help='train/valid/test 데이터셋 저장 경로')

args = parser.parse_args()

def split_user_ids(user_ids, ratios=[7, 1.5, 1.5], seed=42):
    """user_ids를 비율에 맞게 분할하는 함수"""
    random.seed(seed)
    shuffled_ids = user_ids.copy()
    random.shuffle(shuffled_ids)  # 데이터 순서 랜덤화
    
    total = sum(ratios)
    lengths = [int(len(shuffled_ids) * r / total) for r in ratios]
    
    # 길이 보정 (소수점 버림으로 인한 차이)
    diff = len(shuffled_ids) - sum(lengths)
    for i in range(diff):
        lengths[i % len(lengths)] += 1
    
    splits = []
    start = 0
    for length in lengths:
        splits.append(shuffled_ids[start:start+length])
        start += length
    return splits

def mk_datasets(preprocessed_data, user_ids):
    encoder_inputs, decoder_inputs = defaultdict(list), defaultdict(list)
    for uid in user_ids:
        encoder_inputs["question_ids"].append(preprocessed_data[uid]["question_ids"])
        encoder_inputs["concept_ids"].append(preprocessed_data[uid]["concept_ids"])
        encoder_inputs["type_ids"].append(preprocessed_data[uid]["type_ids"])
        encoder_inputs["difficulty_ids"].append(preprocessed_data[uid]["difficulty_ids"])
        encoder_inputs["discriminate_ids"].append(preprocessed_data[uid]["discriminate_ids"])
        decoder_inputs["response_ids"].append(preprocessed_data[uid]["response_ids"])
        decoder_inputs["elapsed_ids"].append(preprocessed_data[uid]["elapsed_ids"])
        decoder_inputs["lag_ids"].append(preprocessed_data[uid]["lag_ids"])
        
    datasets = {
        "encoder_inputs": encoder_inputs,
        "decoder_inputs": decoder_inputs
    }
    
    return datasets

def save_json(save_path, save_name, data):
    with open(os.path.join(save_path, save_name), 'w') as f:
        json.dump(data, f)


if __name__ == "__main__":
    with open(args.data_args_path, 'r') as f:
        data_args = json.load(f)

    start_time = time.time()
    df = pl.read_parquet(args.preprocessed_data_path)
    print(int(time.time() - start_time))
    
    preprocessed_data = {}
    # Iterate over each column (user_id)
    for user_id in tqdm(df.columns):
        # Extract the struct/dict for this user (from the first/only row)
        user_data = df[user_id][0]
        
        # Convert the Polars struct to a Python dictionary
        preprocessed_data[user_id] = {
            "question_ids": user_data["question_ids"],
            "concept_ids": user_data["concept_ids"],
            "type_ids": user_data["type_ids"],
            "difficulty_ids": user_data["difficulty_ids"],
            "discriminate_ids": user_data["discriminate_ids"],
            "response_ids": user_data["response_ids"],
            "elapsed_ids": user_data["elapsed_ids"],
            "lag_ids": user_data["lag_ids"]
        }

    # 사용자 ID 추출
    user_ids = list(preprocessed_data.keys())
    
    # 데이터 분할
    train_ids, valid_ids, test_ids = split_user_ids(
        user_ids, 
        ratios=ratios, 
        seed=42  # 재현성 보장
    )

    user_split_results = {
        "train_ids": train_ids,
        "valid_ids": valid_ids,
        "test_ids": test_ids,
    }
    save_json(args.save_path, "user_split_results.json", user_split_results)

    train_datasets = mk_datasets(preprocessed_data, train_ids)
    valid_datasets = mk_datasets(preprocessed_data, valid_ids)
    test_datasets = mk_datasets(preprocessed_data, test_ids)

    start_time = time.time()
    save_json(args.save_path, "train_datasets.json", train_datasets)
    print(f"train_dataset 저장 완료: {int(time.time()-start_time)}초 소요됨")

    start_time = time.time()
    save_json(args.save_path, "valid_datasets.json", valid_datasets)
    print(f"valid_datasets 저장 완료: {int(time.time()-start_time)}초 소요됨")

    start_time = time.time()
    save_json(args.save_path, "test_datasets.json", test_datasets)
    print(f"test_datasets 저장 완료: {int(time.time()-start_time)}초 소요됨")