# coding : utf-8
# edit : 
# - author : wblee
# - date : 2026-04-30


import os
import json
import argparse
import logging
import polars as pl

from typing import Optional, Dict, Union


parser = argparse.ArgumentParser()
parser.add_argument('--question_meta_path', default='/home/jovyan/work/PICKT/data/Online/Total-Question_Meta.csv', type=str, help="문항 메타 데이터의 파일 경로")
parser.add_argument('--data_args_save_path', default='/home/jovyan/work/PICKT/data/Online/data_args.json', type=str, help="data_args 저장 경로")
args = parser.parse_args()

        
def data_load(args) -> pl.DataFrame():
    """
    풀이이력 데이터를 문항 메타 데이터로 변경하여 load
    """
    # 문항 메타 데이터(`Total-Question_Meta.csv`) load
    question_meta_data = pl.read_csv(args.question_meta_path)
        
    return question_meta_data

def preprocess_data(data):
    """
    텍스트 데이터의 공백 제거
    """
    data = data.with_columns(
        pl.col('type').str.replace_all(r'\s+', ''), # 전체 공백 제거
        pl.col('content').str.replace_all(r'\s+', '') # 전체 공백 제거
    )

    return data

def mk_question2concept(args, meta_data) -> Dict[str, float]:
    """
    문항메타 데이터에서 'question_id'와 'concept'을 맵핑
    """
    question_id_list = meta_data.select('question_id').to_series().to_list()
    
    question2concept={}
    for question_id in question_id_list:
        question2concept[str(question_id)] = meta_data.filter(pl.col('question_id')==question_id)["concept"][0]

    return question2concept
    
def mk_value2id(
    col_name: str, 
    data: pl.DataFrame = None
) -> Dict[str, float]:
    """
    문항메타 데이터에서 'question_id'를 인코딩
    """
    if col_name in ['question_id', 'type', 'activity', 'content', 'concept']:
        if col_name == 'question_id':
            value_list = (data[col_name].unique().to_list())
        else:
            value_list = (data.select(col_name).drop_nulls().unique(maintain_order=True).to_series().to_list())
        value2id = {_name:i for i, _name in enumerate(value_list)}
        
        dict_length = len(value2id)
        value2id["pad_id"] = dict_length
        value2id["unk_id"] = dict_length+1
        
    elif col_name in ['correct', 'difficulty']:
        if col_name == 'correct':
            value2id = {"X": 0, "O": 1, "pad_id": 2, "start_id": 3}
        elif col_name == 'difficulty':
            value2id = {"최하": 0, "하": 1, "중": 2, "상": 3, "최상": 4, "pad_id": 5, "unk_id": 6}

    else:
        raise ValueError(f'Wrong {col_name} name. plaese check this.')
        
    return value2id

def create_data_args(
    args,
    question2id,
    question2concept,
    concept2id, 
    type2id, 
    difficulty2id, 
    content2id, 
    activity2id, 
    response2id
):
    """
    data_args 생성
    """
    data_args = {
        'num_question': len(question2id),
        'num_concept': len(concept2id),
        'num_type': len(type2id),
        'num_difficulty': len(difficulty2id),
        'num_content': len(content2id),
        'num_activity': len(activity2id),
        'num_response': len(response2id),
        'question2id': question2id,
        'question2concept': question2concept,
        'concept2id': concept2id,
        'type2id': type2id,
        'difficulty2id': difficulty2id,
        'content2id': content2id,
        'activity2id': activity2id,
        'response2id': response2id
    }

    with open(args.data_args_save_path, "w", encoding="utf-8") as f:
        json.dump(data_args, f, indent=4, ensure_ascii=False)
    
    print(f"✅ 성공: data_args가 저장되었습니다.")


if __name__ == "__main__":
    meta_data = data_load(args)
    meta_data = preprocess_data(meta_data)
    
    question2concept = mk_question2concept(args, meta_data)
    
    question2id = mk_value2id('question_id', meta_data)
    concept2id = mk_value2id('concept', meta_data)    
    type2id = mk_value2id('type', meta_data)
    difficulty2id = mk_value2id('difficulty', meta_data)
    content2id = mk_value2id('content', meta_data)
    activity2id = mk_value2id('activity', meta_data)
    response2id = mk_value2id(col_name='correct')

    create_data_args(args, question2id, question2concept, concept2id, type2id, difficulty2id, content2id, activity2id, response2id)
    print("Finish..")
