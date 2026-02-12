# coding : utf-8
# edit : 
# - author : lcn
# - date : 2025-


import os
import json
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


# dataset 내 column name 에 따라 아래를 직접 수정해야 됨.
########################################################
quiz_q_text_column_nm = "question_text"
quiz_s_text_column_nm = "explanation"

topic_column_nm = "to_kc"
pre_topic_column_nm = "from_kc"
########################################################

parser = argparse.ArgumentParser(description="Extract question and concept embedding vectors.")
parser.add_argument(
    "--text_type",
    type = str,
    default = "question",
    help = "Determining which text to embed, question or concept."
)
parser.add_argument(
    "--model_path",
    type = str,
    # default = "snumin44/simcse-ko-roberta-supervised",
    default = "sentence-transformers/all-mpnet-base-v2",
    help = "Insert BERT embedding model path in HuggingFace model hub."
)
parser.add_argument(
    "--data_path",
    type = str,
    default = "/home/jovyan/work/repo/datasets/DBE-KT22/question.csv",
    help = "Insert your dataset path."
)
parser.add_argument(
    "--preprocess_data_path",
    type = str,
    default = "/home/jovyan/work/repo/datasets/DBE-KT22/kc.csv",
    help = "Insert your preprocess dataset path."
)
parser.add_argument(
    "--data_args_path",
    type = str,
    default = "/home/jovyan/work/repo/datasets/preprocess/DBE-KT22/data_args.json",
    help = "Insert your dataset arguments json file path."
)
parser.add_argument(
    "--save_tensor_path",
    type = str,
    default = "/home/jovyan/work/repo/datasets/preprocess/DBE-KT22/question_embeddings.pt",
    help = "Insert extracted embedding vector result."
)
parser.add_argument(
    "--max_length",
    type = int,
    default = 512,
    help = "Tokenizer max_length."
)
parser.add_argument(
    "--chunk_size",
    type = int,
    default = 16,
    help = "chunk_size for prevent gpu out of memory."
)
parser.add_argument(
    "--device",
    type = str,
    default = "cuda",
    help = "device."
)

args = parser.parse_args()


def preprocess(args, df, preprocess_df = None):
    if args.text_type == "question":
        df["questionid_encoded"], _ = pd.factorize(df["id"])

    elif args.text_type == "concept":
        id_to_name = preprocess_df.set_index('id')['name']

        # df 에 개념 명 추가
        df["from_kc"] = df["from_knowledgecomponent_id"].map(id_to_name)
        df["to_kc"] = df["to_knowledgecomponent_id"].map(id_to_name)

    return df
    
def ext2question_text(df, data_args):
    # 0. question2id dict 을 value 값 기준으로 sorting
    sorted_question2id = dict(sorted(data_args["question2id"].items(), key=lambda item: item[1]))
    
    # 1. 사전에 데이터 정리 및 매핑 테이블 생성
    quizcode_map = df.set_index("questionid_encoded")[[quiz_q_text_column_nm, quiz_s_text_column_nm]].to_dict("index")
    
    # 2. 순회 시 효율적인 데이터 접근
    full_text = list()
    for expected_id, (quizcode, current_id) in enumerate(sorted_question2id.items()):
        # ID 순서 검증
        if current_id != expected_id:
            raise ValueError("ID 순서가 올바르지 않습니다. 확인해주세요.")

        # 특수 문자 처리
        if quizcode in {"pad_id", "unk_id"}:
            continue
        
        # 데이터 조회
        data = quizcode_map.get(int(quizcode), {})
        q = data.get(quiz_q_text_column_nm, "")
        s = data.get(quiz_s_text_column_nm, None)
        
        # 결과 문자열 생성
        text_entry = f"문제 {q}"
        if s and not pd.isna(s):
            text_entry += f" 해설 {s}"
        full_text.append(text_entry)
    
    return full_text

def ext2concept_text(df, data_args):
    # 0. concept2id dict 을 value 값 기준으로 sorting
    sorted_concept2id = dict(sorted(data_args["concept2id"].items(), key=lambda item: item[1]))
    
    # 1. 순회 시 효율적인 데이터 접근
    full_text = list()
    for expected_id, (topic, current_id) in enumerate(sorted_concept2id.items()):
        # ID 순서 검증
        if current_id != expected_id:
            raise ValueError("ID 순서가 올바르지 않습니다. 확인해주세요.")
    
        # topic이 어느 컬럼에 있는지 확인
        mask = (df[topic_column_nm] == topic)
        if not mask.any():
            mask = (df[pre_topic_column_nm] == topic)

        # 특수 문자 처리
        if topic in {"pad_id", "unk_id"}:
            continue
    
        # # 데이터 조회
        # area = df.loc[mask, area_column_nm].iat[0]
        # depth1 = df.loc[mask, depth1_column_nm].iat[0]
        
        # 결과 문자열 생성
        # text_entry = f"영역 {area} 단원 {depth1} 토픽 {topic}"
        text_entry = f"토픽 {topic}"
        full_text.append(text_entry)
    
    return full_text


if __name__ == "__main__":
    # _, ext = os.path.splitext(args.data_path)
    # ext = ext.lower()
    # if ext == ".csv":
    #     df = pd.read_csv(args.data_path)
    # elif ext in (".xls", ".xlsx"):
    #     df = pd.read_excel(args.data_path)
    
    df = pd.read_csv(args.data_path)
    
    with open(args.data_args_path, "r") as f:
        data_args = json.load(f)

    if args.text_type == "question":
        df = preprocess(args, df)
        full_text = ext2question_text(df, data_args)
        print(f"question2id length: {len(data_args['question2id'])}")
    elif args.text_type == "concept":
        preprocess_df = pd.read_csv(args.preprocess_data_path)
        df = preprocess(args, df, preprocess_df)
        full_text = ext2concept_text(df, data_args)
        print(f"concept2id length: {len(data_args['concept2id'])}")
    else:
        raise ValueError("Please check `text_type`.")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.to(args.device)

    print(f"full_text length: {len(full_text)}")
    output_tensor = torch.tensor([])
    for i in tqdm(range(0, len(full_text), args.chunk_size)):
        chunk = full_text[i:i + args.chunk_size]
        
        encoded_inputs = tokenizer(
            text = chunk,
            add_special_tokens = True,
            padding = 'max_length',
            truncation = True,
            max_length = args.max_length,
            return_tensors = 'pt'
        ).to(args.device)
    
        with torch.no_grad():
            outputs = model(**encoded_inputs, return_dict=True)
        outputs = outputs.pooler_output.cpu()
        output_tensor = torch.concat([output_tensor, outputs])
        
    special_id_tensor = torch.zeros([2, 768])    # pad_id, unk_id 는 0 으로 지정
    output_tensor = torch.concat([output_tensor, special_id_tensor])
    print(f"output_tensor shape: {output_tensor.shape}")
    
    torch.save(output_tensor, args.save_tensor_path)
    print("Finish..")