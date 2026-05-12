# coding : utf-8
# edit : 
# - author : wblee
# - date : 2026-04-30


import os
import time
import json
import argparse
import datetime
import polars as pl

from tqdm.auto import tqdm
from typing import Optional, Dict


parser = argparse.ArgumentParser()
parser.add_argument('--milkt_solved_history_path', default="/home/jovyan/work/PICKT/data/Online/MilkT-RQ1-Solved_History.csv", help="풀이이력 데이터 파일 경로")
parser.add_argument('--data_args_path', default="/home/jovyan/work/PICKT/data/Online/data_args.json", help="data_args 경로")
parser.add_argument('--datasets_save_path', default="/home/jovyan/work/PICKT/data/Online/preprocessed_data.parquet", help="datasets 저장 경로")

args = parser.parse_args()

def data_load(args) -> pl.DataFrame():
    """
    풀이이력 데이터 로드
    """
    # 풀이이력 데이터 load
    solved_history_df = pl.read_csv(args.milkt_solved_history_path)
    
    return solved_history_df

def preprocess_data(data: pl.DataFrame, data_args: dict):

    data = data.with_columns(
        pl.col('type').str.replace_all(" ", ""),
        pl.col('content').str.replace_all(" ", ""),
        pl.col('activity').str.replace_all(" ", ""),
    )
    
    # 1. 전체 데이터 전처리 (벡터화 연산)
    data = data.with_columns(pl.col("question_id").cast(pl.Utf8))
    data = data.sort(["user_id", "credate", "no"]).with_columns(
        # 모든 컬럼 사전 매핑 (벡터화 연산)
        pl.col("question_id").replace_strict(data_args['question2id'], default=data_args['question2id']['unk_id']).alias("question_ids"),
        pl.col("concept").replace_strict(data_args['concept2id'], default=data_args['concept2id']['unk_id']).alias('concept_ids'),
        pl.col("type").replace_strict(data_args['type2id'], default=data_args['type2id']['unk_id']).alias("type_ids"),
        pl.col("difficulty").replace_strict(data_args['difficulty2id'], default=data_args['difficulty2id']['unk_id']).alias("difficulty_ids"),
        pl.col("content").replace_strict(data_args['content2id'], default=data_args['content2id']['unk_id']).alias("content_ids"),
        pl.col("activity").replace_strict(data_args['activity2id'], default=data_args['activity2id']['unk_id']).alias("activity_ids"),
        pl.col("correct").replace_strict(data_args['response2id'], default=-1).alias("response_ids"),
        )
    
    # 2. 그룹별 리스트 변환 (벡터화 집계)
    grouped = data.group_by("user_id").agg(
        pl.col("question_ids"),
        pl.col("concept_ids"),
        pl.col("type_ids"),
        pl.col("difficulty_ids"),
        pl.col("content_ids"),
        pl.col("activity_ids"),
        pl.col("response_ids"),
        )
    
    # 3. 딕셔너리 변환 (진행률 표시)
    preprocessed_data = dict()
    for row in tqdm(grouped.iter_rows(named=True), total=grouped.height, desc="Converting to dict"):
        preprocessed_data[row['user_id']] = {
            'question_ids': row['question_ids'],
            "concept_ids": row["concept_ids"],
            'type_ids': row['type_ids'],
            'difficulty_ids': row['difficulty_ids'],
            'content_ids': row['content_ids'],
            'activity_ids': row['activity_ids'],
            'response_ids': row['response_ids']
        }

    return preprocessed_data



if __name__ == "__main__":
    # Data Load
    data = data_load(args)

    with open(args.data_args_path) as f:
        data_args = json.load(f)

    # 데이터 타입 최적화
    data = data.with_columns(
        pl.col("credate").str.to_datetime(format="%Y-%m-%d %H:%M:%S%.f"),
    )    
    
    # 데이터 처리
    start_time = time.time()
    preprocessed_data = preprocess_data(data, data_args)
    print(f"데이터 전처리 완료: {int(time.time()-start_time)}초 소요됨")

    # 전처리된 데이터 저장
    start_time = time.time()
    data = pl.DataFrame(preprocessed_data)

    data.write_parquet(args.datasets_save_path)
    print(f"전처리 완료된 데이터 저장 완료: {int(time.time()-start_time)}초 소요됨")
    print(f"✅ 성공: preprocessed_data.parquet 이 저장되었습니다.")
