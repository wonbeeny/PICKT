# coding : utf-8
# edit : 
# - author : wblee
# - date : 2025-08-28

import torch

from tabulate import tabulate
from overrides import overrides
from torch_geometric.data import HeteroData
from torch.nn.functional import one_hot

from .base_module import BaseModelModule
from ..utils import pickt_logger
from ..postprocessor import universal_metric, specific_metric


logger = pickt_logger(__name__)

class ModelModule(BaseModelModule):
    @overrides
    def __init__(self, config, model):
        super().__init__(config, model)
        # self.BCEWithLogitsLoss = torch.nn.BCEWithLogitsLoss()    # 시그모이드 + BCELoss를 결합해 loss 연산 최적화
        self.BCELoss = torch.nn.BCELoss()    # 시그모이드 취한 확률값을 input 으로.
    
    @overrides
    def training_step(self, batch, batch_idx):
        """
        각 학습 배치(batch)에서 순전파(forward), 손실(loss) 계산, 로깅 등을 처리.
        모든 학습 배치마다 반복 실행.

        **response 기준 input/output 확인**
        models: [DKT, SAKT, GKT, PICKT, SAINT, ]
            - input: 1~(t-1)
            - output: 2~t
        models: [DKVMN, Dtransformer, ]
            - input: 1~t
            - output: 1~t
        """
        outputs = self._get_outputs(batch)

        labels_flat = batch["labels"][:, 1:].flatten()
        seq_mask = labels_flat != self.config.response2id["pad_id"]
        labels = labels_flat[seq_mask]
        
        if self.config.model_name == "dtransformer":
            loss = outputs.logits
        else:
            if self.config.model_name in ["gkt", "sakt"]:
                predictions = outputs.predictions                
            elif self.config.model_name == "dkt":
                qshft = batch["concept_ids"][:, 1:]
                predictions = (outputs.predictions * one_hot(qshft.long(), self.config.num_concept)).sum(-1)
            elif self.config.model_name in ["pickt", "saint", "akt", "dkvmn"]:
                predictions = outputs.predictions[:, 1:]

            predictions_flat = predictions.flatten()
            predictions = predictions_flat[seq_mask]
            
            loss = self.BCELoss(predictions.float(), labels.float())
            
        log_dict_input = {"train_loss": loss}
        self.log_dict(log_dict_input, sync_dist=True)
        
        self.training_step_outputs.append(loss)
        return loss

    @overrides
    def on_train_epoch_end(self):
        """
        한 에폭(epoch)의 모든 배치 처리 후, 전체 결과를 집계하거나 추가 작업을 수행.
        각 학습 에폭이 끝날 때 한 번 실행.
        """
        avg_loss = torch.stack(self.training_step_outputs).mean()
        log_dict_input = {"train_epoch_loss_mean": avg_loss}
        self.log_dict(log_dict_input, sync_dist=True)
        
        self.training_step_outputs.clear()

    @torch.no_grad()
    @overrides
    def validation_step(self, batch, batch_idx):
        """
        각 검증 배치(batch)에서 모델 추론, 손실 계산 및 메트릭 계산을 수행.
        모든 검증 배치마다 반복 실행.
        """
        outputs = self._get_outputs(batch)
        
        labels_flat = batch["labels"][:, 1:].flatten()
        seq_mask = labels_flat != self.config.response2id["pad_id"]
        labels = labels_flat[seq_mask]
        
        if self.config.model_name == "dtransformer":
            predictions_flat = outputs.predictions[:, 1:].flatten()
            predictions = predictions_flat[seq_mask]

            loss = outputs.logits
        else:
            if self.config.model_name in ["gkt", "sakt"]:
                predictions = outputs.predictions
            elif self.config.model_name == "dkt":
                qshft = batch["concept_ids"][:, 1:]
                predictions = (outputs.predictions * one_hot(qshft.long(), self.config.num_concept)).sum(-1)
            elif self.config.model_name in ["pickt", "saint", "akt", "dkvmn"]:
                predictions = outputs.predictions[:, 1:]

            predictions_flat = predictions.flatten()
            predictions = predictions_flat[seq_mask]

            loss = self.BCELoss(predictions.float(), labels.float())

        log_dict_input = {"valid_loss": loss}
        self.log_dict(log_dict_input, sync_dist=True)

        self.validation_step_outputs.append({"valid_loss": loss, "predictions": predictions, "labels": labels})
        return {"valid_loss": loss, "predictions": predictions, "labels": labels}

    @overrides
    def on_validation_epoch_end(self):
        """
        한 에폭의 모든 검증 배치 처리 후 전체 결과를 집계.
        각 검증 에폭이 끝날 때 한 번 실행.
        """
        avg_loss = torch.stack([x["valid_loss"] for x in self.validation_step_outputs]).mean()

        predictions = torch.cat([x["predictions"] for x in self.validation_step_outputs], dim=0)
        labels = torch.cat([x["labels"] for x in self.validation_step_outputs], dim=0)

        universal_output = universal_metric(self.config, labels, predictions)

        log_dict_input = {
            "valid_epoch_loss_mean": round(float(avg_loss.cpu()), 4),
            "acc_wrong": universal_output.acc_wrong,
            "acc_correct": universal_output.acc_correct,
            "acc_macro": universal_output.acc_macro,
            "acc_micro": universal_output.acc_micro,
            "auc_micro": universal_output.auc_micro,
        }
        self.log_dict(log_dict_input, sync_dist=True)
        table = list(log_dict_input.items())
        logger.info("\n\n" + tabulate(table, headers=["Metric", "Value"], tablefmt="github") + "\n")
        
        self.validation_step_outputs.clear()

    @torch.no_grad()
    @overrides
    def test_step(self, batch, batch_idx):
        """
        각 테스트 배치(batch)에서 모델 추론, 손실 계산 및 메트릭 계산을 수행.
        모든 테스트 배치마다 반복 실행.
        """
        outputs = self._get_outputs(batch)
        
        labels_flat = batch["labels"][:, 1:].flatten()
        seq_mask = labels_flat != self.config.response2id["pad_id"]
        labels = labels_flat[seq_mask]
        
        if self.config.model_name == "dtransformer":
            predictions_flat = outputs.predictions[:, 1:].flatten()
            predictions = predictions_flat[seq_mask]

            loss = outputs.logits
        else:
            if self.config.model_name in ["gkt", "sakt"]:
                predictions = outputs.predictions
            elif self.config.model_name == "dkt":
                qshft = batch["concept_ids"][:, 1:]
                predictions = (outputs.predictions * one_hot(qshft.long(), self.config.num_concept)).sum(-1)
            elif self.config.model_name in ["pickt", "saint", "akt", "dkvmn"]:
                predictions = outputs.predictions[:, 1:]

            predictions_flat = predictions.flatten()
            predictions = predictions_flat[seq_mask]

            loss = self.BCELoss(predictions.float(), labels.float())

        log_dict_input = {"test_loss": loss}
        self.log_dict(log_dict_input, sync_dist=True)

        self.test_step_outputs.append({"test_loss": loss, "predictions": predictions, "labels": labels})
        return {"test_loss": loss, "predictions": predictions, "labels": labels}

    @overrides
    def on_test_epoch_end(self):
        """
        한 에폭의 모든 테스트 배치 처리 후 전체 결과를 집계.
        각 테스트 에폭이 끝날 때 한 번 실행.
        """
        avg_loss = torch.stack([x["test_loss"] for x in self.test_step_outputs]).mean()

        predictions = torch.cat([x["predictions"] for x in self.test_step_outputs], dim=0)
        labels = torch.cat([x["labels"] for x in self.test_step_outputs], dim=0)

        universal_output = universal_metric(self.config, labels, predictions)

        log_dict_input = {
            "test_epoch_loss_mean": round(float(avg_loss.cpu()), 4),
            "acc_wrong": universal_output.acc_wrong,
            "acc_correct": universal_output.acc_correct,
            "acc_macro": universal_output.acc_macro,
            "acc_micro": universal_output.acc_micro,
            "auc_micro": universal_output.auc_micro,
        }
        self.log_dict(log_dict_input, sync_dist=True)
        table = list(log_dict_input.items())
        logger.info("\n\n" + tabulate(table, headers=["Metric", "Value"], tablefmt="github") + "\n")
        
        self.test_step_outputs.clear()

    @torch.no_grad()
    @overrides
    def predict_step(self, batch, batch_idx):
        """
        각 추론 배치(batch)에서 모델 추론 계산을 수행.
        모든 추론 배치마다 반복 실행.
        """
        outputs = self._get_outputs(batch)

        logits = outputs.logits
        predictions = outputs.predictions
        # 직접 결과 처리 또는 예측 결과 수집을 위해 all_gather 수행 
        ### 2개 이상의 gpu 활용한 분산 학습 시 반드시 필요
        gathered_logits = self.all_gather(logits)
        gathered_predictions = self.all_gather(predictions)

        self.predict_step_outputs.append({"logits": gathered_logits, "predictions": gathered_predictions})
        return {"logits": gathered_logits, "predictions": gathered_predictions}

    def _get_outputs(self, batch):
        if self.config.data_name == "milkt":
            if self.config.model_name == "pickt":
                graph_data = HeteroData()
                graph_data['concept'].x = batch["concept_rel_embeds"][0]
                graph_data['question'].x = batch["question_rel_embeds"][0]
                graph_data['concept', 'prereq', 'concept'].edge_index = batch["concept2concept_edge"][0]
                graph_data['concept', 'include', 'question'].edge_index = batch["concept2question_edge"][0]
                
                outputs = self.model(
                    graph_data=graph_data,
                    question_ids=batch["question_ids"],
                    concept_ids=batch["concept_ids"],
                    type_ids=batch["type_ids"],
                    difficulty_ids=batch["difficulty_ids"],
                    discriminate_ids=batch["discriminate_ids"],
                    content_ids=batch["content_ids"],
                    activity_ids=batch["activity_ids"],
                    response_ids=batch["response_ids"],
                    elapsed_ids=batch["elapsed_ids"],
                    lag_ids=batch["lag_ids"],
                    attention_mask=batch["attention_mask"],
                )
            elif self.config.model_name == "dkt":
                outputs = self.model(
                    concept_ids=batch["concept_ids"][:, :-1],
                    response_ids=batch["response_ids"][:, :-1],
                )
            elif self.config.model_name == "gkt":
                graph_data = HeteroData()
                graph_data['concept', 'prereq', 'concept'].edge_index = batch["concept2concept_edge"][0]
                
                outputs = self.model(
                    graph_data=graph_data,
                    concept_ids=batch["concept_ids"],
                    response_ids=batch["response_ids"],
                )
            elif self.config.model_name == "sakt":
                outputs = self.model(
                    q=batch["concept_ids"][:, :-1],
                    r=batch["response_ids"][:, :-1],
                    qry=batch["concept_ids"][:, 1:],
                )
            elif self.config.model_name == "saint":
                outputs = self.model(
                    question_ids=batch["question_ids"],
                    concept_ids=batch["concept_ids"],
                    response_ids=batch["response_ids"],
                    elapsed_ids=batch["elapsed_ids"],
                    lag_ids=batch["lag_ids"],
                )
            elif self.config.model_name == "akt":
                outputs = self.model(
                    q_data=batch["concept_ids"],
                    target=batch["response_ids"],
                    pid_data=batch["question_ids"],
                )
            elif self.config.model_name == "dkvmn":
                outputs = self.model(
                    q=batch["concept_ids"],
                    r=batch["response_ids"],
                )
            elif self.config.model_name == "dtransformer":
                outputs = self.model.get_cl_loss(
                    q=batch["concept_ids"],
                    s=batch["response_ids"],
                    pid=batch["question_ids"],
                    label=batch["response_ids"],
                )
        elif self.config.data_name == "dbekt22":
            if self.config.model_name == "pickt":
                graph_data = HeteroData()
                graph_data['concept'].x = batch["concept_rel_embeds"][0]
                graph_data['question'].x = batch["question_rel_embeds"][0]
                graph_data['concept', 'prereq', 'concept'].edge_index = batch["concept2concept_edge"][0]
                graph_data['concept', 'include', 'question'].edge_index = batch["concept2question_edge"][0]
                
                outputs = self.model(
                    graph_data=graph_data,
                    question_ids=batch["question_ids"],
                    concept_ids=batch["concept_ids"],
                    type_ids=batch["type_ids"],
                    difficulty_ids=batch["difficulty_ids"],
                    discriminate_ids=batch["discriminate_ids"],
                    response_ids=batch["response_ids"],
                    elapsed_ids=batch["elapsed_ids"],
                    lag_ids=batch["lag_ids"],
                    attention_mask=batch["attention_mask"],
                )
            elif self.config.model_name == "dkt":
                outputs = self.model(
                    concept_ids=batch["concept_ids"][:, :-1],
                    response_ids=batch["response_ids"][:, :-1],
                )
            elif self.config.model_name == "gkt":
                graph_data = HeteroData()
                graph_data['concept', 'prereq', 'concept'].edge_index = batch["concept2concept_edge"][0]
                
                outputs = self.model(
                    graph_data=graph_data,
                    concept_ids=batch["concept_ids"],
                    response_ids=batch["response_ids"],
                )
            elif self.config.model_name == "sakt":
                outputs = self.model(
                    q=batch["concept_ids"][:, :-1],
                    r=batch["response_ids"][:, :-1],
                    qry=batch["concept_ids"][:, 1:],
                )
            elif self.config.model_name == "saint":
                outputs = self.model(
                    question_ids=batch["question_ids"],
                    concept_ids=batch["concept_ids"],
                    response_ids=batch["response_ids"],
                    elapsed_ids=batch["elapsed_ids"],
                    lag_ids=batch["lag_ids"],
                )
            elif self.config.model_name == "akt":
                outputs = self.model(
                    q_data=batch["concept_ids"],
                    target=batch["response_ids"],
                    pid_data=batch["question_ids"],
                )
            elif self.config.model_name == "dkvmn":
                outputs = self.model(
                    q=batch["concept_ids"],
                    r=batch["response_ids"],
                )
            elif self.config.model_name == "dtransformer":
                outputs = self.model.get_cl_loss(
                    q=batch["concept_ids"],
                    s=batch["response_ids"],
                    pid=batch["question_ids"],
                    label=batch["response_ids"],
                )

        return outputs