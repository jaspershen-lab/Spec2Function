import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from typing import Optional, Tuple
import torch.nn.functional as F

from .utils import (
    apply_partial_sharing, 
    freeze_encoder_layers, 
    freeze_embedding_layers, 
    get_hidden_size, 
    unfreeze_encoder_layers
)


class ProjectionHead(nn.Module):
    """Simple projection head for contrastive learning"""
    
    def __init__(self, input_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        
        # CLIP-style: Single linear layer with dropout
        self.projection = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_dim, output_dim)
            # NO BatchNorm, NO ReLU!
        )
    
    def forward(self, x):
        return self.projection(x)


class MS2Encoder(nn.Module):
    """MS2谱图编码器（weighted-mean 池化）"""

    def __init__(self, ms_bert, args):
        super().__init__()
        self.ms_bert = ms_bert
        self.hidden_size = get_hidden_size(ms_bert, args.ms_hidden_size)

        # 冻结参数
        if getattr(args, 'freeze_ms_embedding', False):
            freeze_embedding_layers(self.ms_bert)
        if getattr(args, 'freeze_ms_encoder', False):
            freeze_encoder_layers(self.ms_bert, args.freeze_ms_encoder)

        # === FIXED: Simple projection head ===
        embedding_dim = getattr(args, 'embedding_dim', 512)
        dropout = getattr(args, 'projection_dropout', 0.1)
        
        self.projection_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, embedding_dim)
        )
        
        print(f"✅ MS2 Simple projection: {self.hidden_size} -> {embedding_dim}")

    def forward(self, ms_input_id, ms_intensity, ms_attention_mask=None):
        # predict已经返回pooled结果了
        pooled = self.ms_bert.predict(ms_input_id, ms_intensity)  # [B, H]
        
        # 直接project
        projected = self.projection_head(pooled)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized, pooled
    

class TextEncoder(nn.Module):
    """Text encoder with simple projection"""
    
    def __init__(self, text_bert, args):
        super().__init__()
        self.text_bert = text_bert
        self.hidden_size = get_hidden_size(text_bert, args.text_hidden_size)
        
        # 冻结参数
        if hasattr(args, 'freeze_text_embedding') and args.freeze_text_embedding:
            freeze_embedding_layers(self.text_bert)
        
        if hasattr(args, 'freeze_text_encoder') and args.freeze_text_encoder:
            freeze_encoder_layers(self.text_bert, args.freeze_text_encoder)
        
        # 池化策略
        self.pooling_strategy = getattr(args, 'text_pooling', 'cls')
        
        # === FIXED: Simple projection head ===
        embedding_dim = getattr(args, 'embedding_dim', 512)
        dropout = getattr(args, 'projection_dropout', 0.1)
        
        self.projection_head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, embedding_dim)
        )
        
        print(f"✅ Simple projection head: {self.hidden_size} -> {embedding_dim}")
    
    def forward(self, text_input_id, text_attention_mask):
        # 获取BERT输出
        text_outputs = self.text_bert(
            text_input_id, 
            attention_mask=text_attention_mask
        )
        
        # 池化
        if self.pooling_strategy == 'cls':
            text_embeds = text_outputs.last_hidden_state[:, 0, :]
        elif self.pooling_strategy == 'mean':
            hidden_states = text_outputs.last_hidden_state
            attention_mask_expanded = text_attention_mask.unsqueeze(-1).expand(hidden_states.size())
            sum_embeddings = torch.sum(hidden_states * attention_mask_expanded, dim=1)
            sum_mask = torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)
            text_embeds = sum_embeddings / sum_mask
        else:
            raise ValueError(f"Unsupported pooling: {self.pooling_strategy}")
        
        # 简单投影
        projected_embeds = self.projection_head(text_embeds)


        # L2归一化
        normalized_embeds = F.normalize(projected_embeds, p=2, dim=-1)
        
        return normalized_embeds, projected_embeds


class MS2BioText(nn.Module):
    """
    CLIP-style contrastive learning model for MS2-Text.
    Maps MS2 spectra and biological text into a shared embedding space.
    """
    
    def __init__(self, ms_bert, text_bert, args):
        """
        Initialize the CLIP-style contrastive learning model.
        :param ms_bert: Pretrained MS2BERT model
        :param text_bert: Pretrained text model (e.g., BioBERT, HuggingFace BertModel)
        :param args: Configuration arguments
        """
        super().__init__()
        
        # Store the original encoders
        self.ms_bert_original = ms_bert
        self.text_bert_original = text_bert
        
        # Construct encoders
        self.ms_encoder = MS2Encoder(ms_bert, args)
        self.text_encoder = TextEncoder(text_bert, args)
        
        # Embedding dimension
        self.embedding_dim = getattr(args, 'embedding_dim', 512)
        
        # Temperature parameter (used for contrastive learning)
        self.temperature = nn.Parameter(
            torch.tensor(getattr(args, 'temperature', 0.07)),
            requires_grad=getattr(args, 'learnable_temperature', True)
        )
        
        # Whether to use symmetric loss
        self.symmetric_loss = getattr(args, 'symmetric_loss', True)
        
        # MLM head (Masked Language Modeling)
        self.use_mlm = getattr(args, 'use_mlm', False)
        if self.use_mlm:
            self.mlm_head = nn.Linear(self.text_encoder.hidden_size, text_bert.config.vocab_size)
            self.mlm_loss_weight = getattr(args, 'mlm_loss_weight', 0.1)
        
        # MS2 prediction head (auxiliary training task)
        self.use_ms2_prediction = getattr(args, 'use_ms2_prediction', False)
        if self.use_ms2_prediction:
            if not hasattr(args, 'label_columns') or len(args.label_columns) <= 0:
                raise ValueError("If use_ms2_prediction is True, you must provide label_columns with at least one label in the config.")

            self.ms2_prediction_head = nn.Linear(self.ms_encoder.hidden_size, len(args.label_columns))
            self.ms2_prediction_loss_weight = getattr(args, 'ms2_prediction_loss_weight', 0.1)
        self.hparams = vars(args) if hasattr(args, "__dict__") else dict(args)
        self.text_encoder_base = copy.deepcopy(self.text_encoder)
        for p in self.text_encoder_base.parameters():
            p.requires_grad = False

    def compute_distillation_loss(self, text_input_ids, text_attention_mask):
        # teacher：冻结的拷贝，走同样 forward 路径（pool+投影+L2 都包含）
        with torch.no_grad():
            base_embeds, _ = self.text_encoder_base(
                text_input_id=text_input_ids,
                text_attention_mask=text_attention_mask
            )  # [B, D], 已是 L2 归一

        # student：可训练分支
        cur_embeds, _ = self.text_encoder(
            text_input_id=text_input_ids,
            text_attention_mask=text_attention_mask
        )  # [B, D], 已是 L2 归一

        # 1 - cos
        distill_loss = 1.0 - (cur_embeds * base_embeds).sum(dim=-1).mean()
        return distill_loss


    def forward(self, ms_input_id, ms_intensity, text_input_id, text_attention_mask, 
                masked_text_input_id=None, mlm_labels=None, ms2_labels=None,
                hard_neg_input_ids=None, hard_neg_attention_mask=None):
        """
        前向传播
        :return: MS2嵌入, GT文本嵌入, 相似度矩阵, MLM损失（可选）, MS2预测损失（可选）
        """
        # 获取标准化嵌入
        ms_embeds, ms_embeds_raw = self.ms_encoder(ms_input_id, ms_intensity)  # [batch_size, embedding_dim]
        text_embeds, text_hiddenstate = self.text_encoder(text_input_id, text_attention_mask)  # [batch_size, embedding_dim]
        if ms_embeds.dim() == 3 and ms_embeds.size(1) == 1:
            ms_embeds = ms_embeds.squeeze(1)
        # 如果有硬负样本，一起编码
        if hard_neg_input_ids is not None and hard_neg_attention_mask is not None:
            hard_neg_embeds, _ = self.text_encoder(hard_neg_input_ids, hard_neg_attention_mask)
            # 拼接GT和硬负样本
            all_text_embeds = torch.cat([text_embeds, hard_neg_embeds], dim=0)
            similarity_matrix = self.compute_similarity(ms_embeds, all_text_embeds)
        else:
            # 计算相似度矩阵
            similarity_matrix = self.compute_similarity(ms_embeds, text_embeds)
        
        # MLM损失（如果启用且提供了掩码输入）
        mlm_loss = None
        if self.use_mlm and masked_text_input_id is not None and mlm_labels is not None:
            _, mlm_hiddenstate = self.text_encoder(masked_text_input_id, text_attention_mask)
            mlm_loss = self.compute_mlm_loss(mlm_hiddenstate, mlm_labels)
        
        # MS2预测损失
        ms2_prediction_loss = None
        if self.use_ms2_prediction:
            ms2_prediction_loss = self.compute_ms2_prediction_loss(
                ms_embeds_raw, ms2_labels
            )
        
        # 返回ms_embeds和text_embeds（GT的），用于监控collapse
        return ms_embeds, text_embeds, similarity_matrix, mlm_loss, ms2_prediction_loss
    
    def compute_similarity(self, ms_embeds, text_embeds):
        """
        计算MS2和文本嵌入之间的相似度矩阵
        :param ms_embeds: MS2嵌入 [batch_size, embedding_dim]
        :param text_embeds: 文本嵌入 [batch_size, embedding_dim]
        :return: 相似度矩阵 [batch_size, batch_size]
        """
        # 计算余弦相似度并应用温度缩放
        similarity_matrix = torch.matmul(ms_embeds, text_embeds.t()) / self.temperature
        return similarity_matrix
    
    def compute_contrastive_loss(self, similarity_matrix, text_overlap_matrix=None):
        """
        计算对比学习损失，支持多正样本（text overlap）
        :param similarity_matrix: 相似度矩阵 [batch_size, batch_size]
        :param text_overlap_matrix: text overlap矩阵 [batch_size, batch_size], 1表示有共享text
        :return: 对比学习损失
        """
        # 移除多余的维度
        if similarity_matrix.dim() == 3:
            similarity_matrix = similarity_matrix.squeeze(1)
        
        batch_size = similarity_matrix.size(0)
        
        # === 如果没有提供overlap矩阵，使用传统的单正样本loss ===
        if text_overlap_matrix is None:
            labels = torch.arange(batch_size, device=similarity_matrix.device)
            ms_to_text_loss = F.cross_entropy(similarity_matrix, labels)
            
            if self.symmetric_loss:
                text_to_ms_loss = F.cross_entropy(similarity_matrix.t(), labels)
                total_loss = (ms_to_text_loss + text_to_ms_loss) / 2
            else:
                total_loss = ms_to_text_loss
            
            return total_loss
        
        # === 使用text overlap矩阵的多正样本loss ===
        text_overlap_matrix = text_overlap_matrix.to(similarity_matrix.device)
        eye_matrix = torch.eye(batch_size, device=similarity_matrix.device)
        positive_mask = torch.clamp(text_overlap_matrix + eye_matrix, 0, 1)
        # 计算MS2到文本的loss
        ms_to_text_loss = self._compute_multi_positive_loss(
            similarity_matrix, 
            text_overlap_matrix
        )
        
        if self.symmetric_loss:
            # 计算文本到MS2的loss（转置）
            text_to_ms_loss = self._compute_multi_positive_loss(
                similarity_matrix.t(), 
                text_overlap_matrix.t()
            )
            total_loss = (ms_to_text_loss + text_to_ms_loss) / 2
        else:
            total_loss = ms_to_text_loss
        
        return total_loss
    

    def _compute_multi_positive_loss(self, similarity_matrix, positive_mask):
        """
        计算支持多正样本的对比学习loss (修复版)
        :param similarity_matrix: [batch_size, batch_size] - 已经除以temperature
        :param positive_mask: [batch_size, batch_size], 1表示正样本
        """
        batch_size = similarity_matrix.size(0)
        
        # 使用log-sum-exp技巧，数值更稳定
        max_sim = similarity_matrix.max(dim=1, keepdim=True)[0].detach()
        
        # 分子：log(sum(exp(正样本)))
        exp_pos = torch.exp(similarity_matrix - max_sim) * positive_mask
        pos_sum = exp_pos.sum(dim=1)
        log_pos_sum = torch.log(pos_sum + 1e-8) + max_sim.squeeze(1)
        
        # 分母：log(sum(exp(所有样本))) ← 关键修复！
        exp_all = torch.exp(similarity_matrix - max_sim)
        all_sum = exp_all.sum(dim=1)  # ← 不需要mask，包含所有样本！
        log_all_sum = torch.log(all_sum + 1e-8) + max_sim.squeeze(1)
        
        # Loss = -(log(pos) - log(all))
        loss = -(log_pos_sum - log_all_sum)
        
        return loss.mean()
    
    def compute_mlm_loss(self, text_hiddenstate, mlm_labels):
        """
        计算MLM（Masked Language Modeling）损失
        :param masked_text_input_id: 掩码文本输入ID
        :param text_attention_mask: 文本注意力掩码
        :param mlm_labels: MLM标签，-100表示不计算损失的位置
        :return: MLM损失
        """
        
        # 通过MLM头预测词汇
        mlm_logits = self.mlm_head(text_hiddenstate.last_hidden_state)  # [batch_size, seq_len, vocab_size]
        
        # 计算MLM损失
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1),
            ignore_index=-100
        )
        
        return mlm_loss
    
    def compute_ms2_prediction_loss(self,ms_embeds_raw, labels):
        """
        计算MS2预测任务损失
        :param ms_embeds: MS2嵌入
        :return: MS2预测损失
        """

        if len(ms_embeds_raw.shape) == 3:
            ms_embeds_raw = ms_embeds_raw[:, 0, :]  # [batch_size, hidden_size]

        if self.ms2_prediction_head is None:
            return None
        
        # 通过分类头得到logits
        logits = self.ms2_prediction_head(ms_embeds_raw) # [batch_size, num_ms2_classes]
        
        # 计算交叉熵损失
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())
        
        return loss
    
    def get_ms_embeddings(self, ms_input_id, ms_intensity):
        """
        获取MS2嵌入
        :param ms_input_id: MS2输入ID
        :param ms_intensity: MS2强度
        :return: MS2嵌入
        """
        return self.ms_encoder(ms_input_id, ms_intensity)
    
    def get_text_embeddings(self, text_input_id, text_attention_mask):
        """
        获取文本嵌入
        :param text_input_id: 文本输入ID
        :param text_attention_mask: 文本注意力掩码
        :return: 文本嵌入
        """
        text_embeddings , _ =self.text_encoder(text_input_id, text_attention_mask)
        return text_embeddings
    
    def encode_ms(self, ms_input_id, ms_intensity):
        """
        编码MS2谱图（推理时使用）
        :param ms_input_id: MS2输入ID
        :param ms_intensity: MS2强度
        :return: MS2嵌入
        """
        with torch.no_grad():
            return self.get_ms_embeddings(ms_input_id, ms_intensity)
    
    def encode_text(self, text_input_id, text_attention_mask):
        """
        编码文本（推理时使用）
        :param text_input_id: 文本输入ID
        :param text_attention_mask: 文本注意力掩码
        :return: 文本嵌入
        """
        with torch.no_grad():
            return self.get_text_embeddings(text_input_id, text_attention_mask)
    
    def compute_similarity_scores(self, ms_embeds, text_embeds):
        """
        计算相似度分数（推理时使用）
        :param ms_embeds: MS2嵌入 [N, embedding_dim]
        :param text_embeds: 文本嵌入 [M, embedding_dim]
        :return: 相似度分数 [N, M]
        """
        with torch.no_grad():
            # 直接计算余弦相似度，不应用温度缩放
            similarity_scores = torch.matmul(ms_embeds, text_embeds.t())
            return similarity_scores
    
    def unfreeze_model(self, unfreeze_ratio):
        """
        解冻模型参数
        :param unfreeze_ratio: 解冻比例 (0~1之间)
        """
        # 解冻MS2编码器
        unfreeze_encoder_layers(self.ms_encoder.ms_bert, unfreeze_ratio)
        
        # 解冻文本编码器
        unfreeze_encoder_layers(self.text_encoder.text_bert, unfreeze_ratio)
    
    def freeze_encoders(self):
        """冻结预训练编码器，只训练投影头"""
        # 冻结MS2编码器
        for param in self.ms_encoder.ms_bert.parameters():
            param.requires_grad = False
        
        # 冻结文本编码器
        for param in self.text_encoder.text_bert.parameters():
            param.requires_grad = False
    
    def unfreeze_encoders(self):
        """解冻预训练编码器"""
        # 解冻MS2编码器
        for param in self.ms_encoder.ms_bert.parameters():
            param.requires_grad = True
        
        # 解冻文本编码器
        for param in self.text_encoder.text_bert.parameters():
            param.requires_grad = True


# 辅助函数：创建配置示例
def create_clip_config_example():
    """创建CLIP模型配置示例"""
    class CLIPConfig:
        def __init__(self):
            # 基础配置
            self.ms_hidden_size = 768
            self.text_hidden_size = 768
            self.embedding_dim = 512
            
            # 投影头配置
            self.projection_dropout = 0.1
            
            # 文本池化策略
            self.text_pooling = 'cls'  # 'cls', 'mean', 'max'
            
            # 对比学习配置
            self.temperature = 0.07
            self.learnable_temperature = True
            self.symmetric_loss = True
            
            # MLM配置
            self.use_mlm = True
            self.mlm_loss_weight = 0.1
            
            # MS2预测任务配置
            self.use_ms2_prediction = False
            self.ms2_prediction_loss_weight = 0.1
            
            # 冻结配置
            self.freeze_ms_embedding = False
            self.freeze_ms_encoder = 0  # 冻结前N层，0表示不冻结
            self.freeze_text_embedding = False
            self.freeze_text_encoder = 0
    
    return CLIPConfig()


# 辅助函数：创建掩码文本输入
def create_mlm_inputs(text_input_ids, tokenizer, mask_prob=0.15):
    """
    创建MLM训练输入
    :param text_input_ids: 原始文本输入ID
    :param tokenizer: 分词器
    :param mask_prob: 掩码概率
    :return: 掩码后的输入ID和标签
    """
    input_ids = text_input_ids.clone()
    labels = text_input_ids.clone()
    
    # 创建随机掩码（忽略特殊token）
    probability_matrix = torch.full(labels.shape, mask_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # 不计算损失的位置
    
    # 80%的时候用[MASK]替换
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)
    
    # 10%的时候用随机token替换
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]
    
    # 剩余10%保持不变
    
    return input_ids, labels