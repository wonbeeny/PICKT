# coding : utf-8
# edit : 
# - author : lcn
# - date : 2025-05-13


import os
import json
import argparse
import pandas as pd

from typing import Dict, List


parser = argparse.ArgumentParser()

parser.add_argument('--KC_Relations_file_path', default='', help='KC_Relations의 파일 경로')
parser.add_argument('--KCs_file_path', default='', help='KCs의 파일 경로')
parser.add_argument('--Question_KC_Relationships_file_path', default='', help='Question_KC_Relationships의 파일 경로')

parser.add_argument('--data_args_path', default='', help='data_args 경로')
parser.add_argument('--reduced_embeds_path', default='', help='차원 축소된 text embedding 파일 경로')
parser.add_argument('--km_data_save_path', default='', help='km_data 저장 경로')

args = parser.parse_args()

def data_load(args) -> pd.DataFrame():
    """
    문항메타 데이터 & 지식맵 데이터
    로드
    """
    ### KC_Relations 데이터
    kc_df=pd.read_csv(args.KC_Relations_file_path)
    
    ### KCs 데이터
    kc_preprocess_df=pd.read_csv(args.KCs_file_path)
    
    ### Question_KC_Relationships 데이터
    question_df=pd.read_csv(args.Question_KC_Relationships_file_path)

    ### data_args
    with open(args.data_args_path) as f:
        data_args = json.load(f)

    ### text embedding 데이터
    with open(args.reduced_embeds_path, 'r') as f:
        reduced_embeds = json.load(f)
    
    return kc_df, kc_preprocess_df, question_df, data_args, reduced_embeds

def concept2concept_edge(kc_df, data_args, kc_preprocess_df):
    id_to_name = kc_preprocess_df.set_index('id')['name']
    
    kc_df['from_kc'] = kc_df['from_knowledgecomponent_id'].map(id_to_name)
    kc_df['to_kc'] = kc_df['to_knowledgecomponent_id'].map(id_to_name)
    
    concept2concept_edge=[]
    for i in kc_df.index.to_list():
        concept2concept_edge.append(
        {"source":data_args['concept2id'][kc_df.loc[i, 'from_kc']],
         "target":data_args['concept2id'][kc_df.loc[i, 'to_kc']]}
        )

    return concept2concept_edge

def concept2question_edge(question_df, data_args, kc_preprocess_df):
    kc_preprocess_df['kc_name_encoded'], _ = pd.factorize(kc_preprocess_df['name'])
    kc_name_encoded = kc_preprocess_df.set_index('id')['kc_name_encoded']
    id_to_name = kc_preprocess_df.set_index('id')['name']
    
    question_df['kc_name_encoded'] = question_df['knowledgecomponent_id'].map(kc_name_encoded)
    question_df['kc'] = question_df['knowledgecomponent_id'].map(id_to_name)
    
    concept2question_edge=[]
    for i in question_df.index.to_list():
        concept2question_edge.append(
        {"concept":data_args['concept2id'][question_df['kc'][i]],
         "question":data_args['question2id'][str(question_df['question_id'][i])]}
        )
        
    return concept2question_edge


if __name__ == "__main__":
    kc_df, kc_preprocess_df, question_df, data_args, reduced_embeds = data_load(args)
    
    concept_embeds = reduced_embeds['reduced_concept_embeddings']
    question_embeds = reduced_embeds['reduced_question_embeddings']
    concept2concept_edge = concept2concept_edge(kc_df, data_args, kc_preprocess_df)
    concept2question_edge = concept2question_edge(question_df, data_args, kc_preprocess_df)

    km_data = {
        "concept_embeds": concept_embeds,
        "question_embeds": question_embeds,
        "concept2concept_edge": concept2concept_edge,
        "concept2question_edge": concept2question_edge
    }
    
    with open(args.km_data_save_path, 'w') as f:
        json.dump(km_data, f)
    print("Finish..")