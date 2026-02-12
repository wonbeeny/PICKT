# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-05-19

from typing import Optional, Dict
from dataclasses import dataclass, field


@dataclass
class MilktDataArguments:
    """Arguments MilkT Datasets."""

    num_question: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of questions contain pad_id and unk_id. 문항 수 + PAD_id, UNK_id"
        }
    )
    num_concept: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of concepts contain pad_id and unk_id. 개념 수 + PAD_id, UNK_id"
        }
    )
    num_type: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of question types contain pad_id and unk_id. 문항유형 수 + PAD_id, UNK_id"
        }
    )
    num_difficulty: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of question difficulties contain pad_id and unk_id. 난이도 카테고리 수 + PAD_id, UNK_id"
        }
    )
    num_discriminate: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of question discriminates contain pad_id and unk_id. 변별도 카테고리 수 + PAD_id, UNK_id"
        }
    )
    num_content: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of question contents contain pad_id and unk_id. 내용영역 수 + PAD_id, UNK_id"
        }
    )
    num_activity: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of question activities contain pad_id and unk_id. 행동영역 수 + PAD_id, UNK_id"
        }
    )
    num_response: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of student responses contain pad_id. 학생의 정오답 여부 수 + PAD_id"
        }
    )
    num_elapsed: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of student's time it took to solve a question contain pad_id and unk_id. 문항풀이 시간 카테고리 수 + PAD_id, UNK_id"
        }
    )
    num_lag: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of student's time it took to solve one question and move on to the next one contain pad_id and unk_id. 문항을 푼 이후부터 다음 문항 풀시 시작하는데까지 걸린 시간 카테고리 수 + PAD_id, UNK_id"
        }
    )
    question2id: Dict[str, int] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for question number to sequence id containing pad_id and unk_id. dictionary for 문항 번호 : id."
        }
    )
    question2concept: Dict[str, str] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for question number to concept text without special id. dictionary for 문항 번호 : 개념명."
        }
    )
    concept2id: Dict[str, int] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for concept text to sequence id containing pad_id. dictionary for 개념명 : id."
        }
    )
    type2id: Dict[str, int] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for question type text to sequence id containing pad_id and unk_id. dictionary for 문항 유형 str : id."
        }
    )
    difficulty2id: Dict[str, int] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for question difficulty text to sequence id containing pad_id and unk_id. dictionary for 난이도 str : id."
        }
    )
    discriminate2id: Dict[str, int] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for question discriminate text to sequence id containing pad_id and unk_id. dictionary for 변별도 str : id."
        }
    )
    content2id: Dict[str, int] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for question content text to sequence id containing pad_id and unk_id. dictionary for 내용영역 str : id."
        }
    )
    activity2id: Dict[str, int] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for question activity text to sequence id containing pad_id and unk_id. dictionary for 행동영역 str : id."
        }
    )
    response2id: Dict[str, int] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for student responses text to sequence id containing pad_id. dictionary for 정오답 str : id."
        }
    )
    elapsed2id: Dict[str, int] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for student elapsed time category text to sequence id containing pad_id and unk_id. dictionary for 문제풀이 시간 범주 str : id."
        }
    )
    lag2id: Dict[str, int] = field(
        default=None,
        metadata={
            "help": "Mapping dictionary for student lag time category text to sequence id containing pad_id and unk_id. dictionary for 다음 문제푸는데 걸린 시간 범주 str : id."
        }
    )

    def __post_init__(self):
        pass