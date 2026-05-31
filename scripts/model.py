"""
模型定义：
- BertBinaryClassifier：基于 BERT 的二分类器（Stage1 和 Stage2 共用）
- CascadeSentimentModel：级联推理模型
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertPreTrainedModel


class BertBinaryClassifier(BertPreTrainedModel):
    """
    BERT + 分类头的二分类模型
    用于 Stage1（中性 vs 非中性）和 Stage2（负面 vs 正面）
    """

    def __init__(self, config, dropout: float = 0.1):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
    ):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        # 取 [CLS] 向量
        pooled = outputs[1]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)  # (batch, 1)

        loss = None
        if labels is not None:
            loss_fn = nn.BCEWithLogitsLoss()
            loss = loss_fn(logits.view(-1), labels.float())

        return {"loss": loss, "logits": logits}


class CascadeSentimentModel:
    """
    级联推理模型：
    Stage1: 判断是否中性
      → 是中性 → 输出 1（中性）
      → 非中性 → 进入 Stage2 判断正面/负面
    """

    def __init__(self, stage1_model: BertBinaryClassifier, stage2_model: BertBinaryClassifier):
        self.stage1 = stage1_model
        self.stage2 = stage2_model

    def eval(self):
        self.stage1.eval()
        self.stage2.eval()

    def to(self, device):
        self.stage1.to(device)
        self.stage2.to(device)
        return self

    def predict_stage1(self, inputs: dict) -> torch.Tensor:
        """Stage1: 返回预测概率 (batch,)"""
        with torch.no_grad():
            outputs = self.stage1(**inputs)
            probs = torch.sigmoid(outputs["logits"].view(-1))
        return probs

    def predict_stage2(self, inputs: dict) -> torch.Tensor:
        """Stage2: 返回预测概率 (batch,)"""
        with torch.no_grad():
            outputs = self.stage2(**inputs)
            probs = torch.sigmoid(outputs["logits"].view(-1))
        return probs

    def predict(self, inputs: dict, threshold: float = 0.5) -> list:
        """
        级联预测，返回三分类结果。
        返回: list of int, 0=负面, 1=中性, 2=正面
        
        pipeline:
          Stage1 prob >= threshold → 中性(1)
          Stage1 prob < threshold → Stage2
            Stage2 prob >= threshold → 正面(2)
            Stage2 prob < threshold → 负面(0)
        """
        stage1_probs = self.predict_stage1(inputs)

        predictions = []
        # 找出非中性的样本索引
        non_neutral_mask = stage1_probs < threshold
        non_neutral_indices = torch.where(non_neutral_mask)[0]

        # 默认全部预测为中性
        preds = torch.ones(len(stage1_probs), dtype=torch.long, device=stage1_probs.device)

        if len(non_neutral_indices) > 0:
            # 取非中性样本进入 Stage2
            non_neutral_inputs = {
                k: v[non_neutral_indices] for k, v in inputs.items()
            }
            stage2_probs = self.predict_stage2(non_neutral_inputs)
            # 正面=2, 负面=0
            stage2_preds = (stage2_probs >= threshold).long() * 2
            preds[non_neutral_indices] = stage2_preds

        return preds.cpu().tolist()

    def predict_with_probs(self, inputs: dict, threshold: float = 0.5) -> dict:
        """
        返回三分类结果 + 各阶段概率
        """
        stage1_probs = self.predict_stage1(inputs)
        predictions = []
        final_probs = []  # 三分类概率

        non_neutral_mask = stage1_probs < threshold
        non_neutral_indices = torch.where(non_neutral_mask)[0]

        stage2_probs_all = torch.zeros(len(stage1_probs), device=stage1_probs.device)

        if len(non_neutral_indices) > 0:
            non_neutral_inputs = {
                k: v[non_neutral_indices] for k, v in inputs.items()
            }
            s2_probs = self.predict_stage2(non_neutral_inputs)
            stage2_probs_all[non_neutral_indices] = s2_probs

        # 构建三分类概率: [P(负面), P(中性), P(正面)]
        for i in range(len(stage1_probs)):
            p_neutral = stage1_probs[i].item()
            if p_neutral >= threshold:
                # 是中性
                predictions.append(1)
                final_probs.append([(1 - p_neutral) / 2, p_neutral, (1 - p_neutral) / 2])
            else:
                p_pos = stage2_probs_all[i].item()
                if p_pos >= threshold:
                    predictions.append(2)
                    final_probs.append([(1 - p_neutral) * (1 - p_pos), p_neutral, (1 - p_neutral) * p_pos])
                else:
                    predictions.append(0)
                    final_probs.append([(1 - p_neutral) * (1 - p_pos), p_neutral, (1 - p_neutral) * p_pos])

        return {"predictions": predictions, "probs": final_probs}
