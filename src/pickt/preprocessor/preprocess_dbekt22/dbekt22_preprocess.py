# coding : utf-8
# edit : 
# - author : lcn
# - date : 2025-


import os
import time
import json
import argparse
import datetime
import numpy as np
import pandas as pd

from tqdm.auto import tqdm
from typing import Optional, Dict


parser = argparse.ArgumentParser()
parser.add_argument('--log_file_path', default='', help="log 파일 경로")
parser.add_argument('--question_choice_file_path', default='', help="Question_Choices 파일 경로")
parser.add_argument('--birt_file_path', default='', help="BIRT 파일 경로")

parser.add_argument('--km_data_path', default='', help="km_data 경로")
parser.add_argument('--data_args_path', default='', help="data_args 경로")
parser.add_argument('--datasets_save_path', default='', help="datasets 저장 경로")

args = parser.parse_args()

args = parser.parse_args()

def data_load(args) -> pd.DataFrame():
    log = pd.read_csv(args.log_file_path)
    question_choices = pd.read_csv(args.question_choice_file_path)
    birt = pd.read_csv(args.birt_file_path)
    
    with open(args.data_args_path) as f:
        data_args = json.load(f)
    
    with open(args.km_data_path) as f:
        km_data = json.load(f)
    
    question2concept_dict = dict(zip(pd.DataFrame(km_data['concept2question_edge'])['question'], pd.DataFrame(km_data['concept2question_edge'])['concept']))
    
    question_type = {}
    for question_id in question_choices['question_id'].unique():
        if len(question_choices[question_choices['question_id'] == question_id]) == 2:
            question_type[int(question_id)] = '2지 선택'
        elif len(question_choices[question_choices['question_id'] == question_id]) == 3:
            question_type[int(question_id)] = '3지 선택'
        elif len(question_choices[question_choices['question_id'] == question_id]) == 4:
            question_type[int(question_id)] = '4지 선택'
        elif len(question_choices[question_choices['question_id'] == question_id]) == 5:
            question_type[int(question_id)] = '5지 선택'
        else:
            print(question_id)
    
    
    log = log[~log.duplicated(subset=['student_id','question_id','start_time'])]#[['start_time', 'end_time','answer_state','student_id','question_id']]
    
    log['answer_state'] = log['answer_state'].map({True: 'O', False: 'X'})
    
    log['new_start_time'] = pd.to_datetime(log['start_time'], errors='coerce', utc=True)
    log['new_end_time'] = pd.to_datetime(log['end_time'],errors='coerce', utc=True)
    
    log = log.sort_values(['student_id','new_start_time'])
    
    # elapsed time은 문제를 마친 시간 - 시작한 시간 = 문제를 푸는 데 걸린 시간
    log['elapsed_time'] = (log['new_end_time'] - log['new_start_time']).dt.total_seconds()
    
    # elapsed time이 음수인 경우는 Nan으로 변경, 총 4건 존재 
    log['elapsed_time'] = log['elapsed_time'].where((log['elapsed_time'].isna())|(log['elapsed_time'] >= 0), np.nan)
    
    # lag time은 다음 문제 시작 시간 - 전 문제 마친 시간 
    # 학생별 첫번째 행의 lag_time 값은 Nan으로
    log['lag_time'] = (
        log.groupby('student_id')['new_end_time'].shift(1) - log['new_start_time']
    ).dt.total_seconds() 
    
    # lag time의 음수값을 양수로 변경(코드상 '전 문제 마친 시간 - 다음 문제 시작 시간'이기 때문 )
    log['lag_time'] = log['lag_time'].apply(lambda x: -x if pd.notna(x) else x)
    
    # lag time이 음수라면 Nan으로 값 변경
    log['lag_time'] = log['lag_time'].where((log['lag_time'].isna()) | (log['lag_time'] >= 0), np.nan)
    
    log = pd.merge(log, birt, how = 'left', on = 'question_id')
    log['type'] = log['question_id'].map(question_type)
    log['questionid_encoded'] = log['question_id'].astype(str).map(data_args['question2id'])
    log['concept_ids'] = log['questionid_encoded'].map(question2concept_dict)
    
    
    log = log[['student_id', 'new_start_time', 'new_end_time', 'answer_state', 'type', 'questionid_encoded', 'concept_ids', 'lag_time', 'elapsed_time', 'difficulty_category', 'distcrimination_category']]
    
    return log, data_args

def get_elapsed_ids(elapsed2id, x):
    try:
        elapsedtime= float(x)
        if 0 <= elapsedtime < 5:
            res = elapsed2id['0s 이상 ~ 5s 미만']
        elif 5 <= elapsedtime < 10:
            res = elapsed2id['5s 이상 ~ 10s 미만']
        elif 10 <= elapsedtime < 15:
            res = elapsed2id['10s 이상 ~ 15s 미만']
        elif 15 <= elapsedtime < 20:
            res = elapsed2id['15s 이상 ~ 20s 미만']
        elif 20 <= elapsedtime < 30:
            res = elapsed2id['20s 이상 ~ 30s 미만']
        elif 30 <= elapsedtime < 40:
            res = elapsed2id['30s 이상 ~ 40s 미만']
        elif 40 <= elapsedtime < 50:
            res = elapsed2id['40s 이상 ~ 50s 미만']
        elif 50 <= elapsedtime < 60:
            res = elapsed2id['50s 이상 ~ 60s 미만']
        elif 60 <= elapsedtime < 90:
            res = elapsed2id['60s 이상 ~ 90s 미만']
        elif 90 <= elapsedtime < 120:
            res = elapsed2id['90s 이상 ~ 120s 미만s']
        elif 120 <= elapsedtime < 1800:
            res = elapsed2id['120s 이상']
        else:
            res = elapsed2id['unk_id']
    except:
        res = elapsed2id['unk_id']

    return res

def get_lag_ids(lag2id, x):
    try:
        lagtime= float(x)
        if 0 <= lagtime < 1:
            res = lag2id["0s 이상 ~ 1s 미만"]
        elif 1 <= lagtime < 2:
            res = lag2id["1s 이상 ~ 2s 미만"]
        elif 2 <= lagtime < 3:
            res = lag2id["2s 이상 ~ 3s 미만"]
        elif 3 <= lagtime < 4:
            res = lag2id["3s 이상 ~ 4s 미만"]
        elif 4 <= lagtime < 5:
            res = lag2id["4s 이상 ~ 5s 미만"]
        elif 5 <= lagtime < 10:
            res = lag2id["5s 이상 ~ 10s 미만"]
        elif 10 <= lagtime < 20:
            res = lag2id["10s 이상 ~ 20s 미만"]
        elif 20 <= lagtime < 30:
            res = lag2id["20s 이상 ~ 30s 미만"]
        elif 30 <= lagtime < 60:
            res = lag2id["30s 이상 ~ 60s 미만"]
        elif 60 <= lagtime:
            res = lag2id["60s 이상"]
        else:
            res = lag2id['unk_id']
    except:
        res = lag2id['unk_id']

    return res

def preprocess_data(data: pd.DataFrame, data_args: dict):
    # 1. 데이터 정렬
    data = data.sort_values(["student_id", "new_start_time", "questionid_encoded"])
    
    # 2. 딕셔너리 매핑 + unknown 값 처리
    data["question_ids"]     = data["questionid_encoded"].astype(str).map(data_args["question2id"]).fillna(data_args["question2id"]["unk_id"]).astype(int)
    data["type_ids"]         = data["type"].map(data_args["type2id"]).fillna(data_args["type2id"]["unk_id"]).astype(int)
    data["difficulty_ids"]   = data["difficulty_category"].map(data_args["difficulty2id"]).fillna(data_args["difficulty2id"]["unk_id"]).astype(int)
    data["discriminate_ids"] = data["distcrimination_category"].map(data_args["discriminate2id"]).fillna(data_args["discriminate2id"]["unk_id"]).astype(int)
    data["response_ids"]     = data["answer_state"].map(data_args["response2id"]).fillna(-1).astype(int)
    
    # 3. 사용자 정의 함수 적용
    data["elapsed_ids"] = data["elapsed_time"].apply(lambda x: get_elapsed_ids(data_args["elapsed2id"], x))
    data["lag_ids"]     = data["lag_time"].apply(lambda x: get_lag_ids(data_args["lag2id"], x))

    grouped = data.groupby("student_id").agg({
        "question_ids": list,
        "concept_ids": list,
        "type_ids": list,
        "difficulty_ids": list,
        "discriminate_ids": list,
        "response_ids": list,
        "elapsed_ids": list,
        "lag_ids": list
    }).reset_index()

    preprocessed_data = dict()

    for _, row in tqdm(grouped.iterrows(), total=len(grouped), desc="Converting to dict"):
        preprocessed_data[row["student_id"]] = {
            "question_ids": row["question_ids"],
            "concept_ids": row["concept_ids"],
            "type_ids": row["type_ids"],
            "difficulty_ids": row["difficulty_ids"],
            "discriminate_ids": row["discriminate_ids"],
            "response_ids": row["response_ids"],
            "elapsed_ids": row["elapsed_ids"],
            "lag_ids": row["lag_ids"]
        }

    data = pd.DataFrame()
    for student_id in preprocessed_data.keys():
        student_data = pd.DataFrame({str(student_id) : [preprocessed_data[student_id]]})
        data = pd.concat([data, student_data], axis = 1)
        
    return data


if __name__ == "__main__":
    # Data Load
    start_time = time.time()
    data, data_args = data_load(args)
    print(f"데이터 Load 완료: {int(time.time()-start_time)}초 소요됨")

    # 데이터 처리
    start_time = time.time()
    data = preprocess_data(data, data_args)
    print(f"데이터 전처리 완료: {int(time.time()-start_time)}초 소요됨")

    # 전처리된 데이터 저장
    start_time = time.time()
    data.to_parquet(args.datasets_save_path)
    print(f"전처리 완료된 데이터 저장 완료: {int(time.time()-start_time)}초 소요됨")