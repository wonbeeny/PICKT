# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-06-23

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class PicktMilktModelArguments:
    """Arguments Pickt Model."""
    
    han_metadata: Optional[tuple] = field(
        default=(
            ['concept', 'question'], 
            [('concept', 'prereq', 'concept'), ('concept', 'include', 'question')]
        ),
        metadata={
            "help": "concept and question relation metadata."
        },
    )
    han_in_channels: Optional[int] = field(
        default=64,
        metadata={
            "help": "han input size of the Pickt model."
        }
    )
    han_hidden_channels: Optional[int] = field(
        default=128,
        metadata={
            "help": "han hidden size of the Pickt model."
        }
    )
    han_num_heads: Optional[int] = field(
        default=4,
        metadata={
            "help": "Number of han attention heads for each attention layer in the GNN-HAN."
        }
    )
    max_position_embeddings: Optional[int] = field(
        default=256,
        metadata={
            "help": "The maximum sequence length that this model might ever be used with. Typically set this to something large just in case (e.g., 128 or 256 or user defined value)."
        }
    )
    hidden_size: Optional[int] = field(
        default=256,
        metadata={
            "help": "Dimensionality of the encoder and decoder layers. Should be a multiple of num_attention_heads."
        }
    )
    intermediate_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": "Dimensionality of the `intermediate` (often named feed-forward) layer in the Transformer encoder and decoder. Should be a hidden_size*4"
        }
    )
    num_attention_heads: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of attention heads for each attention layer in the Transformer encoder and decoder."
        }
    )
    num_encoder_layers: Optional[int] = field(
        default=4,
        metadata={
            "help": "Number of hidden layers in the Transformer encoder."
        }
    )
    num_decoder_layers: Optional[int] = field(
        default=4,
        metadata={
            "help": "Number of hidden layers in the Transformer decoder."
        }
    )
    hidden_dropout_prob: Optional[float] = field(
        default=0,
        metadata={
            "help": "The dropout probability for all fully connected layers in the embeddings, encoder and decoder."
        }
    )
    attention_probs_dropout_prob: Optional[float] = field(
        default=0,
        metadata={
            "help": "The dropout ratio for the attention probabilities."
        }
    )
    layer_norm_eps: Optional[float] = field(
        default=1e-12,
        metadata={
            "help": "The epsilon used by the layer normalization layers."
        }
    )
    chunk_size_feed_forward: Optional[int] = field(
        default=0,
        metadata={
            "help": "the dimension over which the input_tensors should be chunked. 0으로 설정 시 피드포워드 레이어가 청크 분할 없이 전체 입력을 한 번에 처리."
        }
    )
    hidden_act: Optional[str] = field(
        default="gelu",
        metadata={
            "help": "The non-linear activation function (function or string) in the all layers. If string, `gelu`, `relu`, `silu` and `gelu_new` are supported."
        }
    )
    position_embedding_type: Optional[str] = field(
        default="absolute",
        metadata={
            "help": "Type of position embedding. Choose one of `absolute`, `relative_key`, `relative_key_query`. For positional embeddings use `absolute`. For more information on `relative_key`, please refer to [Self-Attention with Relative Position Representations (Shaw et al.)](https://arxiv.org/abs/1803.02155). For more information on `relative_key_query`, please refer to *Method 4* in [Improve Transformer Models with Better Relative Position Embeddings (Huang et al.)](https://arxiv.org/abs/2009.13658)."
        }
    )
    output_attentions: Optional[bool] = field(
        default=False,
        metadata={
            "help": ""
        }
    )
    output_hidden_states: Optional[bool] = field(
        default=False,
        metadata={
            "help": ""
        }
    )
    use_return_dict: Optional[bool] = field(
        default=False,
        metadata={
            "help": ""
        }
    )

    def __post_init__(self):
        pass


@dataclass
class SaintMilktModelArguments:
    """Arguments Pickt Model."""

    hidden_size: Optional[int] = field(
        default=256,
        metadata={
            "help": "Dimensionality of the encoder and decoder layers. Should be a multiple of num_attention_heads."
        }
    )
    num_attention_heads: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of attention heads for each attention layer in the Transformer encoder and decoder."
        }
    )
    num_encoder_layers: Optional[int] = field(
        default=4,
        metadata={
            "help": "Number of hidden layers in the Transformer encoder."
        }
    )
    num_decoder_layers: Optional[int] = field(
        default=4,
        metadata={
            "help": "Number of hidden layers in the Transformer decoder."
        }
    )
    attention_probs_dropout_prob: Optional[float] = field(
        default=0,
        metadata={
            "help": "The dropout ratio for the attention probabilities."
        }
    )


@dataclass
class DktMilktModelArguments:
    """Arguments DKT Model."""

    hidden_size: Optional[int] = field(
        default=256,
        metadata={
            "help": "Dimensionality of the hidden layers."
        }
    )
    n_dims : Optional[int] = field(
        default=0, # paper default check
        metadata={
            "help": "Dimensionality of the hidden layers."
        }
    )
    

@dataclass
class GktMilktModelArguments:
    """Arguments GKT Model."""

    hidden_dim: Optional[int] = field(
        default=32,
        metadata={
            "help": "Dimension of hidden knowledge states."
        }
    )
    embedding_dim: Optional[int] = field(
        default=32,
        metadata={
            "help": "Dimension of concept embedding."
        }
    )
    dropout: Optional[float] = field(
        default=0.5,
        metadata={
            "help": "Dropout rate (1 - keep probability)."
        }
    )
    bias: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether to add bias for neural network layers."
        }
    )
    binary: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether only use 0/1 for results."
        }
    )
    has_cuda: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Whether only use 0/1 for results."
        }
    )
    edge_type_num: Optional[float] = field(
        default=2,
        metadata={
            "help": "The number of edge types to infer."
        }
    )
    graph_type: Optional[str] = field(
        default="Dense",
        metadata={
            "help": "The type of latent concept graph."
        }
    )
    graph_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "The type of latent concept graph."
        }
    )

@dataclass
class SaktMilktModelArguments:
    """Arguments SAKT Model."""

    num_attention_heads: Optional[int] = field(
        default=5,
        metadata={
            "help": "Number of attention heads for each attention layer in the Transformer encoder and decoder."
        }
    )
    dropout: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "The dropout ratio for the attention probabilities."
        }
    )
    emb_size: Optional[float] = field(
        default=200, # option: [50, 100, 150, 200]
        metadata={
            "help": "Dimension of embedding layer"
        }
    )



@dataclass
class AktMilktModelArguments:
    """Arguments AKT Model."""

    hidden_size: Optional[int] = field(
        default=256,
        metadata={
            "help": "Dimensionality of the encoder and decoder layers. Should be a multiple of num_attention_heads."
        }
    )
    num_attention_heads: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of attention heads for each attention layer in the Transformer encoder and decoder."
        }
    )
    intermediate_size: Optional[int] = field(
        default=1024,
        metadata={
            "help": "feed-forward dimension."
        }
    )
    final_fc_dim: Optional[int] = field(
        default=512,
        metadata={
            "help": "Final fc dimension."
        }
    )
    n_blocks: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of transformers layers."
        }
    )
    dropout: Optional[float] = field(
        default=0.05,
        metadata={
            "help": "The dropout ratio for the attention probabilities."
        }
    )
    separate_qa: Optional[bool] = field(
        default=False,
        metadata={
            "help": "use response_ids like dkt response_ids or not."
        }
    )


@dataclass
class DkvmnMilktModelArguments:
    """Arguments DKVMN Model."""

    dim_s: Optional[int] = field(
        default=50,
        metadata={
            "help": "the dimension of the state vectors in this model."
        }
    )
    size_m: Optional[int] = field(
        default=20,
        metadata={
            "help": "the memory size of this model."
        }
    )


@dataclass
class DTransformerMilktModelArguments:
    """Arguments DTransformer Model."""

    hidden_size: Optional[int] = field(
        default=128,
        metadata={
            "help": "Dimensionality of the encoder and decoder layers. Should be a multiple of num_attention_heads."
        }
    )
    num_attention_heads: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of attention heads for each attention layer in the Transformer encoder and decoder."
        }
    )
    intermediate_size: Optional[int] = field(
        default=256,
        metadata={
            "help": "feed-forward dimension."
        }
    )
    n_layers: Optional[int] = field(
        default=3,
        metadata={
            "help": "Number of transformers layers."
        }
    )
    n_know: Optional[int] = field(
        default=32,
        metadata={
            "help": "dimension of knowledge parameter."
        }
    )
    dropout: Optional[float] = field(
        default=0.2,
        metadata={
            "help": "The dropout ratio for the attention probabilities."
        }
    )
    hard_neg: Optional[bool] = field(
        default=False,
        metadata={
            "help": "use hard negative samples in CL."
        }
    )