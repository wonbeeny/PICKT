# coding : utf-8
# edit : 
# - author : wblee
# - date : 2026-04-30


import os
import json
import argparse
import polars as pl

from typing import Dict, List


parser = argparse.ArgumentParser()

parser.add_argument('--knowledge_map_file_path', default="/home/jovyan/work/PICKT/data/Online/MilkT-Knowledge_Map.csv", help='선후 관계 연결된 지식맵 파일 경로')
parser.add_argument('--question_meta_path', default='/home/jovyan/work/PICKT/data/Online/Total-Question_Meta.csv', type=str, help="문항 메타 데이터의 파일 경로")
parser.add_argument('--data_args_path', default='/home/jovyan/work/PICKT/data/Online/data_args.json', help='data_args 경로')
parser.add_argument('--reduced_embeds_path', default="/home/jovyan/work/PICKT/data/Online/reduced_embeddings_pca.json", help='차원 축소된 text embedding 파일 경로')
parser.add_argument('--km_data_save_path', default="/home/jovyan/work/PICKT/data/Online/km_data.json", help='km_data 저장 경로')


args = parser.parse_args()

def data_load(args) -> pl.DataFrame():
    """
    문항메타 데이터 & 지식맵 데이터
    로드
    """
    # 지식맵 데이터 load
    map_data = pl.read_csv(args.knowledge_map_file_path)

    # 문항 메타 데이터(`Total-Question_Meta.csv`) load
    question_meta_data = pl.read_csv(args.question_meta_path)

    # data_args load
    with open(args.data_args_path) as f:
        data_args = json.load(f)

    # text embedding 데이터 load
    with open(args.reduced_embeds_path, 'r') as f:
        reduced_embeds = json.load(f)
    
    return map_data, question_meta_data, data_args, reduced_embeds


def concept2concept_edge(map_data, data_args) -> list:
    
    data_unique = map_data.unique()
    data_unique = data_unique.with_row_index(name='index_columns') # 코드 수정 : map_data -> data_unique

    concept2concept=[]
    for i in data_unique['index_columns']:
        if data_unique["from_concept"][i] == None:
            continue
        else:
            source = data_args['concept2id'][data_unique["from_concept"][i]]
        
        target = data_args['concept2id'][data_unique["to_concept"][i]]
            
        concept2concept.append({
            "source": source, "target": target
        })

    return concept2concept

def concept2question_edge(meta_data, data_args) -> list:
    meta_data = meta_data.with_row_index(name='index_columns')

    concept2question=[]
    for i in meta_data['index_columns']:
        concept2question.append({
            "concept": data_args['concept2id'][meta_data['concept'][i]], 
            "question": data_args['question2id'][str(meta_data['question_id'][i])]
        })

    return concept2question


if __name__ == "__main__":
    map_data, meta_data, data_args, reduced_embeds = data_load(args)

    concept_embeds = reduced_embeds['reduced_concept_embeddings']
    question_embeds = reduced_embeds['reduced_question_embeddings']
    concept2concept = concept2concept_edge(map_data, data_args)
    concept2question = concept2question_edge(meta_data, data_args)
    
    km_data = {
        "concept_embeds": concept_embeds,
        "question_embeds": question_embeds,
        "concept2concept_edge": concept2concept,
        "concept2question_edge": concept2question
    }

    with open(args.km_data_save_path, 'w') as f:
        json.dump(km_data, f)
    
    print(f"✅ 성공: km_data.json 이 저장되었습니다.")
    print("Finish..")