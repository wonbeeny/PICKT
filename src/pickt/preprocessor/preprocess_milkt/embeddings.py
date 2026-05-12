# coding : utf-8
# edit : 
# - author : wblee
# - date : 2026-04-30


import os
import json
import torch
import argparse
import pandas as pd

from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

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
    default = "snumin44/simcse-ko-roberta-supervised",
    help = "Insert BERT embedding model path in HuggingFace model hub."
)
parser.add_argument(
    "--question_text_data_path",
    type = str,
    default = "/home/jovyan/work/PICKT/data/Online/MilkT-Question_Text.csv",
    help = "Insert your dataset path."
)
parser.add_argument(
    "--data_args_path",
    type = str,
    default = "/home/jovyan/work/PICKT/data/Online/data_args.json",
    help = "Insert your dataset arguments json file path."
)
parser.add_argument(
    "--save_tensor_path",
    type = str,
    default = "/home/jovyan/work/PICKT/data/Online/question_embeddings.pt",
    help = "Insert extracted embedding vector result."
)
parser.add_argument(
    "--max_length",
    type = int,
    default = 512,
    help = "Tokenizer max_length."
)
parser.add_argument(
    "--batch_size",
    type = int,
    default = 16,
    help = "mini batch size for prevent gpu out of memory."
)
parser.add_argument(
    "--device",
    type = str,
    default = "cuda",
    help = "device."
)

args = parser.parse_args()

def clean_text(text):
    if text:
        return text.replace("\displaystyle", "").replace("\quad", "").replace("\,", "")
    return text
    
def ext2question_text(df, data_args):
    # 0. question2id dict 을 value 값 기준으로 sorting
    sorted_question2id = dict(sorted(data_args["question2id"].items(), key=lambda item: item[1]))
    
    # 1. 사전에 데이터 정리 및 매핑 테이블 생성
    quizcode_map = df.set_index("question_id")[['question_txt', 'explain_txt']].applymap(
        lambda x: clean_text(x) if pd.notna(x) else x
    ).to_dict("index")
    
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
        q = data.get('question_txt', "")
        s = data.get('explain_txt', None)
    
        # 결과 문자열 생성
        if q and not pd.isna(q):
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
        mask = (df['concept'] == topic)
    
        # 특수 문자 처리
        if topic in {"pad_id", "unk_id"}:
            continue

        area = df.loc[mask, 'content'].iat[0]
        area = area.replace(" ", "")
        topic_name = df.loc[mask, 'concept'].iat[0]
        topic_description = df[df['concept'] == topic_name]['description'].item()
        
        text_entry = f"영역 {area} 단원 {topic_name} 단원 description {topic_description}"
        
        full_text.append(text_entry)
        
    return full_text

if __name__ == "__main__":
    df = pd.read_csv(args.question_text_data_path)

    with open(args.data_args_path, "r") as f:
        data_args = json.load(f)
    
    if args.text_type == "question":
        full_text = ext2question_text(df, data_args)
        print(f"question2id length: {len(data_args['question2id'])}")
    elif args.text_type == "concept":
        full_text = ext2concept_text(df, data_args)
        print(f"concept2id length: {len(data_args['concept2id'])}")
    else:
        raise ValueError("Please check `text_type`.")
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModel.from_pretrained(args.model_path)
    model.to(args.device)

    # print(f"full_text length: {len(full_text)}")
    output_tensor = torch.tensor([])
    for i in tqdm(range(0, len(full_text), args.batch_size)):
        chunk = full_text[i:i + args.batch_size]
        
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
    print(f"✅ 성공: {args.text_type}_embedding.pt 가 저장되었습니다.")
    print("Finish..")