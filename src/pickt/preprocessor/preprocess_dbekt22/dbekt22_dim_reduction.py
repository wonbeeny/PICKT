# coding : utf-8
# edit : 
# - author : lcn
# - date : 2025-


import os
import json
import torch
import argparse

from umap import UMAP
from sklearn.decomposition import PCA


parser = argparse.ArgumentParser(description="Extract question and concept embedding vectors.")
parser.add_argument(
    "--question_embed_path",
    type = str,
    default = "/home/jovyan/work/repo/datasets/preprocess/DBE-KT22/question_embeddings.pt",
    help = "Path to the file containing the question embeddings (e.g., BERT-encoded vectors for each question)."
)
parser.add_argument(
    "--concept_embed_path",
    type = str,
    default = "/home/jovyan/work/repo/datasets/preprocess/DBE-KT22/concept_embeddings.pt",
    help = "Path to the file containing the concept embeddings (e.g., BERT-encoded vectors for each concept or tag)."
)
parser.add_argument(
    "--n_components",
    type = int,
    default = 64,
    help = "The number of dimensions to reduce the embeddings to."
)
parser.add_argument(
    "--dr_type",
    type = str,
    default = "pca",
    help = "The dimensionality reduction method to use. Choose either `pca` for Principal Component Analysis or `umap` for Uniform Manifold Approximation and Projection."
)
parser.add_argument(
    "--save_path",
    type = str,
    default = "/home/jovyan/work/repo/datasets/preprocess/DBE-KT22",
    help = "Path to save the reduced-dimension embeddings as a file."
)
args = parser.parse_args()

def dim_red_pca(embed, n_components):
    """
    PCA는 UMAP에 비해 하이퍼파라미터가 적고, 성능에 큰 영향을 주는 옵션도 제한적.
    대부분의 상황에서 추가로 조정할 옵션은 거의 없음.
    PCA를 1순위로 생각하고 있음. 
    """
    pca = PCA(n_components=n_components, random_state=42, svd_solver='full')
    reduced_pca = pca.fit_transform(embed)
    reduced_pca[0] = 0    # pad_id 위치는 전부 0으로 변환
    reduced_pca[1] = 0    # unk_id 위치는 전부 0으로 변환
    print(f"설명된 분산 비율 합계: {sum(pca.explained_variance_ratio_):.2f}")
    
    return reduced_pca

def dim_red_umap(embed, n_components):
    """
    사용자 설정 파라미터를 수학 개념 텍스트 임베딩 최적화 관점에서 설정.
    최적화 관점은 perplexity 에게 조언을 받아 적용하였음.
    """
    umap_model = UMAP(
        n_components=n_components,
        n_neighbors=35,          # 기본값 15 → 35
        min_dist=0.2,            # 기본값 0.1 → 0.2
        metric='cosine',         # 텍스트 임베딩에 최적화
        spread=1.5,              # 기본값 1.0 → 1.5
        local_connectivity=2,    # 기본값 1 → 2
        random_state=42,         # 재현성 보장
        n_jobs=-1                # 멀티코어 활용
    )
    reduced_umap = umap_model.fit_transform(embed)
    reduced_umap[0] = 0    # pad_id 위치는 전부 0으로 변환
    reduced_umap[1] = 0    # unk_id 위치는 전부 0으로 변환
    
    return reduced_umap


if __name__ == "__main__":
    q_embed = torch.load(args.question_embed_path)
    c_embed = torch.load(args.concept_embed_path)

    if args.dr_type == "pca":
        q_reduced = dim_red_pca(q_embed, args.n_components)
        c_reduced = dim_red_pca(c_embed, args.n_components)
    elif args.dr_type == "umap":
        q_reduced = dim_red_umap(q_embed, args.n_components)
        c_reduced = dim_red_umap(c_embed, args.n_components)

    reduced_embeds = {
        "reduced_question_embeddings": q_reduced.tolist(),
        "reduced_concept_embeddings": c_reduced.tolist(),
    }

    with open(args.save_path, "w", encoding="utf-8") as output_file:
        json.dump(reduced_embeds, output_file, indent=4, ensure_ascii=False)
    print("Finish..")