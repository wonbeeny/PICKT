# coding : utf-8
# edit : 
# - author : wblee
# - date : 2026-04-29

import torch

from lightning.pytorch.callbacks import ModelCheckpoint, Callback

class PredictionCollector(Callback):
    def __init__(self):
        self.all_preds = list()
        self.all_responses = list()
        self.all_questions = list()

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self.all_preds.append(outputs)
        self.all_responses.append(batch['labels'])    # batch['response_ids'] 로 수정 예정
        if batch.get('question_ids') is None:
            self.all_questions.append(batch['concept_ids'])
        else:
            self.all_questions.append(batch['question_ids'])

    def on_predict_end(self, trainer, pl_module):
        # # 모든 프로세스가 여기까지 도달할 때까지 대기
        # trainer.strategy.barrier()

        # logits = torch.cat([x["logits"].squeeze(-1) for x in self.all_preds], dim=0)
        predictions = torch.cat([x["predictions"].squeeze(-1) for x in self.all_preds], dim=0)
        responses = torch.cat([x for x in self.all_responses], dim=0)
        questions = torch.cat([x for x in self.all_questions], dim=0)

        # 결과 활용 (예: 변수에 할당)
        self.final_output = {
            # "logits": logits.tolist(),
            "predictions": predictions.tolist(),
            "responses": responses.tolist(),
            "questions": questions.tolist()
        }