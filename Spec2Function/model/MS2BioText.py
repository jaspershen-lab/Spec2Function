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
    """MS2 spectrum encoder (weighted-mean pooling)."""

    def __init__(self, ms_bert, args):
        super().__init__()
        self.ms_bert = ms_bert
        self.hidden_size = get_hidden_size(ms_bert, args.ms_hidden_size)

        # Freeze parameters
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
        # predict already returns the pooled result
        pooled = self.ms_bert.predict(ms_input_id, ms_intensity)  # [B, H]

        # Project directly
        projected = self.projection_head(pooled)
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized, pooled
    

class TextEncoder(nn.Module):
    """Text encoder with simple projection"""
    
    def __init__(self, text_bert, args):
        super().__init__()
        self.text_bert = text_bert
        self.hidden_size = get_hidden_size(text_bert, args.text_hidden_size)
        
        # Freeze parameters
        if hasattr(args, 'freeze_text_embedding') and args.freeze_text_embedding:
            freeze_embedding_layers(self.text_bert)

        if hasattr(args, 'freeze_text_encoder') and args.freeze_text_encoder:
            freeze_encoder_layers(self.text_bert, args.freeze_text_encoder)

        # Pooling strategy
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
        # Get BERT output
        text_outputs = self.text_bert(
            text_input_id,
            attention_mask=text_attention_mask
        )

        # Pooling
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
        
        # Simple projection
        projected_embeds = self.projection_head(text_embeds)


        # L2 normalization
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
        # teacher: frozen copy, takes the same forward path (pool + projection + L2)
        with torch.no_grad():
            base_embeds, _ = self.text_encoder_base(
                text_input_id=text_input_ids,
                text_attention_mask=text_attention_mask
            )  # [B, D], already L2-normalized

        # student: trainable branch
        cur_embeds, _ = self.text_encoder(
            text_input_id=text_input_ids,
            text_attention_mask=text_attention_mask
        )  # [B, D], already L2-normalized

        # 1 - cos
        distill_loss = 1.0 - (cur_embeds * base_embeds).sum(dim=-1).mean()
        return distill_loss


    def forward(self, ms_input_id, ms_intensity, text_input_id, text_attention_mask,
                masked_text_input_id=None, mlm_labels=None, ms2_labels=None,
                hard_neg_input_ids=None, hard_neg_attention_mask=None):
        """
        Forward pass.
        :return: MS2 embedding, GT text embedding, similarity matrix, MLM loss (optional), MS2 prediction loss (optional)
        """
        # Get normalized embeddings
        ms_embeds, ms_embeds_raw = self.ms_encoder(ms_input_id, ms_intensity)  # [batch_size, embedding_dim]
        text_embeds, text_hiddenstate = self.text_encoder(text_input_id, text_attention_mask)  # [batch_size, embedding_dim]
        if ms_embeds.dim() == 3 and ms_embeds.size(1) == 1:
            ms_embeds = ms_embeds.squeeze(1)
        # Encode hard negatives together if provided
        if hard_neg_input_ids is not None and hard_neg_attention_mask is not None:
            hard_neg_embeds, _ = self.text_encoder(hard_neg_input_ids, hard_neg_attention_mask)
            # Concatenate GT and hard negatives
            all_text_embeds = torch.cat([text_embeds, hard_neg_embeds], dim=0)
            similarity_matrix = self.compute_similarity(ms_embeds, all_text_embeds)
        else:
            # Compute similarity matrix
            similarity_matrix = self.compute_similarity(ms_embeds, text_embeds)

        # MLM loss (if enabled and masked inputs are provided)
        mlm_loss = None
        if self.use_mlm and masked_text_input_id is not None and mlm_labels is not None:
            _, mlm_hiddenstate = self.text_encoder(masked_text_input_id, text_attention_mask)
            mlm_loss = self.compute_mlm_loss(mlm_hiddenstate, mlm_labels)

        # MS2 prediction loss
        ms2_prediction_loss = None
        if self.use_ms2_prediction:
            ms2_prediction_loss = self.compute_ms2_prediction_loss(
                ms_embeds_raw, ms2_labels
            )

        # Return ms_embeds and (GT) text_embeds for collapse monitoring
        return ms_embeds, text_embeds, similarity_matrix, mlm_loss, ms2_prediction_loss

    def compute_similarity(self, ms_embeds, text_embeds):
        """
        Compute the similarity matrix between MS2 and text embeddings.
        :param ms_embeds: MS2 embeddings [batch_size, embedding_dim]
        :param text_embeds: text embeddings [batch_size, embedding_dim]
        :return: similarity matrix [batch_size, batch_size]
        """
        # Compute cosine similarity with temperature scaling
        similarity_matrix = torch.matmul(ms_embeds, text_embeds.t()) / self.temperature
        return similarity_matrix

    def compute_contrastive_loss(self, similarity_matrix, text_overlap_matrix=None):
        """
        Compute the contrastive loss, with optional multi-positive support (text overlap).
        :param similarity_matrix: similarity matrix [batch_size, batch_size]
        :param text_overlap_matrix: text overlap matrix [batch_size, batch_size]; 1 means shared text
        :return: contrastive loss
        """
        # Remove extra dimensions
        if similarity_matrix.dim() == 3:
            similarity_matrix = similarity_matrix.squeeze(1)

        batch_size = similarity_matrix.size(0)

        # === If no overlap matrix is provided, use the standard single-positive loss ===
        if text_overlap_matrix is None:
            labels = torch.arange(batch_size, device=similarity_matrix.device)
            ms_to_text_loss = F.cross_entropy(similarity_matrix, labels)

            if self.symmetric_loss:
                text_to_ms_loss = F.cross_entropy(similarity_matrix.t(), labels)
                total_loss = (ms_to_text_loss + text_to_ms_loss) / 2
            else:
                total_loss = ms_to_text_loss

            return total_loss

        # === Use the multi-positive loss with the text overlap matrix ===
        text_overlap_matrix = text_overlap_matrix.to(similarity_matrix.device)
        eye_matrix = torch.eye(batch_size, device=similarity_matrix.device)
        positive_mask = torch.clamp(text_overlap_matrix + eye_matrix, 0, 1)
        # Compute MS2-to-text loss
        ms_to_text_loss = self._compute_multi_positive_loss(
            similarity_matrix,
            text_overlap_matrix
        )

        if self.symmetric_loss:
            # Compute text-to-MS2 loss (transpose)
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
        Compute a multi-positive contrastive loss (fixed version).
        :param similarity_matrix: [batch_size, batch_size] - already divided by temperature
        :param positive_mask: [batch_size, batch_size], 1 marks positives
        """
        batch_size = similarity_matrix.size(0)

        # Use the log-sum-exp trick for numerical stability
        max_sim = similarity_matrix.max(dim=1, keepdim=True)[0].detach()

        # Numerator: log(sum(exp(positives)))
        exp_pos = torch.exp(similarity_matrix - max_sim) * positive_mask
        pos_sum = exp_pos.sum(dim=1)
        log_pos_sum = torch.log(pos_sum + 1e-8) + max_sim.squeeze(1)

        # Denominator: log(sum(exp(all samples)))  <-- the key fix!
        exp_all = torch.exp(similarity_matrix - max_sim)
        all_sum = exp_all.sum(dim=1)  # <-- no mask: include all samples!
        log_all_sum = torch.log(all_sum + 1e-8) + max_sim.squeeze(1)

        # Loss = -(log(pos) - log(all))
        loss = -(log_pos_sum - log_all_sum)

        return loss.mean()

    def compute_mlm_loss(self, text_hiddenstate, mlm_labels):
        """
        Compute the MLM (Masked Language Modeling) loss.
        :param masked_text_input_id: masked text input IDs
        :param text_attention_mask: text attention mask
        :param mlm_labels: MLM labels; -100 marks positions excluded from the loss
        :return: MLM loss
        """

        # Predict vocabulary via the MLM head
        mlm_logits = self.mlm_head(text_hiddenstate.last_hidden_state)  # [batch_size, seq_len, vocab_size]

        # Compute MLM loss
        mlm_loss = F.cross_entropy(
            mlm_logits.view(-1, mlm_logits.size(-1)),
            mlm_labels.view(-1),
            ignore_index=-100
        )

        return mlm_loss

    def compute_ms2_prediction_loss(self,ms_embeds_raw, labels):
        """
        Compute the MS2 prediction task loss.
        :param ms_embeds: MS2 embeddings
        :return: MS2 prediction loss
        """

        if len(ms_embeds_raw.shape) == 3:
            ms_embeds_raw = ms_embeds_raw[:, 0, :]  # [batch_size, hidden_size]

        if self.ms2_prediction_head is None:
            return None

        # Get logits via the classification head
        logits = self.ms2_prediction_head(ms_embeds_raw) # [batch_size, num_ms2_classes]

        # Compute cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(logits, labels.float())

        return loss

    def get_ms_embeddings(self, ms_input_id, ms_intensity):
        """
        Get MS2 embeddings.
        :param ms_input_id: MS2 input IDs
        :param ms_intensity: MS2 intensities
        :return: MS2 embeddings
        """
        return self.ms_encoder(ms_input_id, ms_intensity)

    def get_text_embeddings(self, text_input_id, text_attention_mask):
        """
        Get text embeddings.
        :param text_input_id: text input IDs
        :param text_attention_mask: text attention mask
        :return: text embeddings
        """
        text_embeddings , _ =self.text_encoder(text_input_id, text_attention_mask)
        return text_embeddings

    def encode_ms(self, ms_input_id, ms_intensity):
        """
        Encode an MS2 spectrum (used at inference time).
        :param ms_input_id: MS2 input IDs
        :param ms_intensity: MS2 intensities
        :return: MS2 embeddings
        """
        with torch.no_grad():
            return self.get_ms_embeddings(ms_input_id, ms_intensity)

    def encode_text(self, text_input_id, text_attention_mask):
        """
        Encode text (used at inference time).
        :param text_input_id: text input IDs
        :param text_attention_mask: text attention mask
        :return: text embeddings
        """
        with torch.no_grad():
            return self.get_text_embeddings(text_input_id, text_attention_mask)

    def compute_similarity_scores(self, ms_embeds, text_embeds):
        """
        Compute similarity scores (used at inference time).
        :param ms_embeds: MS2 embeddings [N, embedding_dim]
        :param text_embeds: text embeddings [M, embedding_dim]
        :return: similarity scores [N, M]
        """
        with torch.no_grad():
            # Compute cosine similarity directly, without temperature scaling
            similarity_scores = torch.matmul(ms_embeds, text_embeds.t())
            return similarity_scores

    def unfreeze_model(self, unfreeze_ratio):
        """
        Unfreeze model parameters.
        :param unfreeze_ratio: unfreezing ratio (between 0 and 1)
        """
        # Unfreeze MS2 encoder
        unfreeze_encoder_layers(self.ms_encoder.ms_bert, unfreeze_ratio)

        # Unfreeze text encoder
        unfreeze_encoder_layers(self.text_encoder.text_bert, unfreeze_ratio)

    def freeze_encoders(self):
        """Freeze pretrained encoders and only train projection heads."""
        # Freeze MS2 encoder
        for param in self.ms_encoder.ms_bert.parameters():
            param.requires_grad = False

        # Freeze text encoder
        for param in self.text_encoder.text_bert.parameters():
            param.requires_grad = False

    def unfreeze_encoders(self):
        """Unfreeze pretrained encoders."""
        # Unfreeze MS2 encoder
        for param in self.ms_encoder.ms_bert.parameters():
            param.requires_grad = True

        # Unfreeze text encoder
        for param in self.text_encoder.text_bert.parameters():
            param.requires_grad = True


# Helper: create a configuration example
def create_clip_config_example():
    """Create an example CLIP model configuration."""
    class CLIPConfig:
        def __init__(self):
            # Base configuration
            self.ms_hidden_size = 768
            self.text_hidden_size = 768
            self.embedding_dim = 512

            # Projection head
            self.projection_dropout = 0.1

            # Text pooling strategy
            self.text_pooling = 'cls'  # 'cls', 'mean', 'max'

            # Contrastive learning
            self.temperature = 0.07
            self.learnable_temperature = True
            self.symmetric_loss = True

            # MLM
            self.use_mlm = True
            self.mlm_loss_weight = 0.1

            # MS2 prediction
            self.use_ms2_prediction = False
            self.ms2_prediction_loss_weight = 0.1

            # Freezing
            self.freeze_ms_embedding = False
            self.freeze_ms_encoder = 0  # freeze first N layers; 0 means no freezing
            self.freeze_text_embedding = False
            self.freeze_text_encoder = 0

    return CLIPConfig()


# Helper: create masked text input
def create_mlm_inputs(text_input_ids, tokenizer, mask_prob=0.15):
    """
    Create MLM training inputs.
    :param text_input_ids: original text input IDs
    :param tokenizer: tokenizer
    :param mask_prob: mask probability
    :return: masked input IDs and labels
    """
    input_ids = text_input_ids.clone()
    labels = text_input_ids.clone()

    # Create the random mask (skip special tokens)
    probability_matrix = torch.full(labels.shape, mask_prob)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # positions excluded from the loss

    # 80% of the time: replace with [MASK]
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    input_ids[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time: replace with a random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
    input_ids[indices_random] = random_words[indices_random]

    # Remaining 10%: leave unchanged

    return input_ids, labels