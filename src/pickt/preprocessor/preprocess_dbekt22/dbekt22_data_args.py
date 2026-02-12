import argparse
import pandas as pd
import numpy as np
import json
from typing import List

parser = argparse.ArgumentParser()

parser.add_argument('--Question_KC_Relationships_path', default='/home/jovyan/work/repo/datasets/DBE-KT22/Question_KC_Relationships.csv', type=str, help="KC 문항 매핑 파일 경로")
parser.add_argument('--KCs_path', default='/home/jovyan/work/repo/datasets/DBE-KT22/KCs.csv', type=str, help="KC 파일 경로")
parser.add_argument('--Questions_path', default='/home/jovyan/work/repo/datasets/DBE-KT22/Questions.csv', type=str, help="문항 파일 경로")
parser.add_argument('--Transaction_path', default='', type=str, help="풀이이력 파일 경로")
parser.add_argument('--Question_Choices_path', default='/home/jovyan/work/repo/datasets/DBE-KT22/Question_Choices.csv', type=str, help="문항 개념 매핑 경로")
parser.add_argument('--data_args_save_path', default='/home/jovyan/work/repo/datasets/preprocessed/DBE-KT22/data_args.json', type=str, help="data_args 저장 경로")
args = parser.parse_args()


## 풀이이력 데이터 불러온 후
## 필요한 칼럼만 남기고
## 'student_id','question_id','start_time' 중복인 경우 제외
## answer_state 칼럼 전처리
## elapsed_time, lag_time 산출 후 라벨링 진행

def preprocess(log: pd.DataFrame) -> pd.DataFrame:
    
    log = pd.read_csv(args.Transaction_path)
    log = log[~log.duplicated(subset=['student_id','question_id','start_time'])]#[['start_time', 'end_time','answer_state','student_id','question_id']]

    #log = log 
    
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


    elapsed_bins = [0, 5, 10, 15, 20, 30, 40, 50, 60, 90, 120, float('inf')]
    elapsed_labels = ['0s 이상 ~ 5s 미만', '5s 이상 ~ 10s 미만', '10s 이상 ~ 15s 미만', '15s 이상 ~ 20s 미만', '20s 이상 ~ 30s 미만', 
                      '30s 이상 ~ 40s 미만', '40s 이상 ~ 50s 미만', '50s 이상 ~ 60s 미만',
                      '60s 이상 ~ 90s 미만','90s 이상 ~ 120s 미만', '120s 이상']
    
    log['elapsed_group'] = pd.cut(log['elapsed_time'], bins=elapsed_bins, labels=elapsed_labels, right=False)
    
    
    lag_bins = [0,1,2,3,4,5, 10, 20, 30, 60, float('inf')]
    lag_labels = ['0s 이상 ~ 1s 미만','1s 이상 ~ 2s 미만','2s 이상 ~ 3s 미만','3s 이상 ~ 4s 미만',
                  '4s 이상 ~ 5s 미만','5s 이상 ~ 10s 미만', '10s 이상 ~ 20s 미만', '20s 이상 ~ 30s 미만',
                  '30s 이상 ~ 60s 미만','60s 이상']
    
    log['lag_group'] = pd.cut(log['lag_time'], bins=lag_bins, labels=lag_labels, right=False)

    return log

    
if __name__ == "__main__":
    
    question_kc = pd.read_csv(args.Question_KC_Relationships_path)
    kc = pd.read_csv(args.KCs_path)
    question = pd.read_csv(args.Questions_path)
    log = pd.read_csv(args.Transaction_path)
    question_choice = pd.read_csv(args.Question_Choices_path)
    
    log = preprocess(log)

#######################################################################################################    
    kc['kc_name_encoded'], _ = pd.factorize(kc['name'])
    concept2id = dict(zip(kc['name'], kc['kc_name_encoded']))
    concept2id = {**concept2id,
                  'pad_id': len(concept2id.keys()),
                  'unk_id' : len(concept2id.keys()) + 1}
    
#######################################################################################################    
    question['questionid_encoded'], _ = pd.factorize(question['id'])
    question2id = dict(zip(question['id'].astype(str), question['questionid_encoded']))
    question2id = {**question2id,
                   'pad_id': len(question2id.keys()),
                   'unk_id' : len(question2id.keys())+1}
    
########################################################################################################   
    ## 기존의 문항ID를 새 문항ID와 조인
    id_to_question_encoded = question.set_index('id')['questionid_encoded']
    
    question_kc['questionid_encoded'] = question_kc['question_id'].map(id_to_question_encoded)
    
    id_to_name = kc.set_index('id')['name']
    
    ## 기존의 개념ID를 개념명과 조인 후
    ## 이어서 개념명을 새 개념ID와 조인
    question_kc['kc'] = question_kc['knowledgecomponent_id'].map(id_to_name)
    
    question_kc = question_kc.drop_duplicates(['questionid_encoded', 'kc'])
    
    question2concept={}
    for i in question_kc.index:
        question2concept[int(question_kc.iloc[i]['questionid_encoded'])] = question_kc.iloc[i]['kc']

#########################################################################################################
    ## type2id
    type_list = []
    
    for question_id in question_choice['question_id'].unique():
        if len(question_choice[question_choice['question_id'] == question_id]) == 2:
            type_list.append('2지 선택')
        elif len(question_choice[question_choice['question_id'] == question_id]) == 3:
            type_list.append('3지 선택')
        elif len(question_choice[question_choice['question_id'] == question_id]) == 4:
            type_list.append('4지 선택')
        elif len(question_choice[question_choice['question_id'] == question_id]) == 5:
            type_list.append('5지 선택')
        #else:
        #    print(question_id)
    
    list(set(type_list))
    
    type2id={}
    for i, question_type in enumerate(list(set(type_list))):
        type2id[question_type] = i
    
    type2id['pad_id'] = len(list(set(type_list)))+1
    type2id['unk_id'] = len(list(set(type_list)))+2
    
######################################################################################################
    
    response2id={}
    
    response2id['X']=0
    response2id['O']=1
    response2id['pad_id']=2
    response2id['start_id']=3
    
###################################################################################################
    elapsed2id = dict()
    elapsed_labels = ['0s 이상 ~ 5s 미만', '5s 이상 ~ 10s 미만', '10s 이상 ~ 15s 미만', '15s 이상 ~ 20s 미만', '20s 이상 ~ 30s 미만', 
                      '30s 이상 ~ 40s 미만', '40s 이상 ~ 50s 미만', '50s 이상 ~ 60s 미만',
                      '60s 이상 ~ 90s 미만','90s 이상 ~ 120s 미만', '120s 이상']
    
    for i, label in enumerate(elapsed_labels):
        elapsed2id[label] = i
    
    elapsed2id['pad_id'] = len(elapsed_labels)
    elapsed2id['unk_id'] = len(elapsed_labels)+1
    elapsed2id['start_id'] = len(elapsed_labels)+2
    
 #################################################################################################
    
    lag2id = dict()
    lag_labels = ['0s 이상 ~ 1s 미만','1s 이상 ~ 2s 미만','2s 이상 ~ 3s 미만','3s 이상 ~ 4s 미만',
                  '4s 이상 ~ 5s 미만','5s 이상 ~ 10s 미만', '10s 이상 ~ 20s 미만', '20s 이상 ~ 30s 미만',
                  '30s 이상 ~ 60s 미만','60s 이상']
    for i, label in enumerate(lag_labels):
        lag2id[label] = i
    
    lag2id['pad_id'] = len(lag_labels)
    lag2id['unk_id'] = len(lag_labels)+1
    lag2id['start_id'] = len(lag_labels)+2
    
#####################################################################################################
    
    difficulty2id = {"very easy": 0, "easy": 1, "medium": 2, "hard": 3, "very hard": 4, "pad_id": 5, "unk_id": 6}

    discriminate2id = {"none": 0, "low": 1, "moderate": 2, "high": 3, "perfect": 4, "pad_id": 5, "unk_id": 6}

#####################################################################################################

    data_args = {
        'num_question': len(question2id),
        'num_concept': len(concept2id),
        'num_type': len(type2id),
        'num_difficulty': len(difficulty2id),
        'num_discriminate': len(discriminate2id),
        #'num_content': len(content2id),
        #'num_activity': len(activity2id),
        'num_response': len(response2id),
        'num_elapsed': len(elapsed2id),
        'num_lag': len(lag2id),
        'question2id': question2id,
        'question2concept': question2concept,
        'concept2id': concept2id,
        'type2id': type2id,
        'difficulty2id': difficulty2id,
        'discriminate2id': discriminate2id,
        #'content2id': content2id,
        #'activity2id': activity2id,
        'response2id': response2id,
        'elapsed2id': elapsed2id,
        'lag2id': lag2id
    }

    with open(args.data_args_save_path, "w", encoding="utf-8") as f:
        json.dump(data_args, f, indent=4, ensure_ascii=False)
#####################################################################################################
