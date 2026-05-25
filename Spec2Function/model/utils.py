import copy
import torch.nn as nn

def apply_partial_sharing(ms_bert, unshared_layers):
    """
    Share the first n layers and keep the last `unshared_layers` independent.

    Args:
        ms_bert: original MSBERT encoder
        unshared_layers: number of layers (int) to train independently

    Returns:
        Encoder with partially shared parameters.
    """
    try:
        # Create the new encoder
        target_model = copy.deepcopy(ms_bert)  # copy the entire model first

        # Get transformer blocks from source and target models
        source_transformer_blocks = ms_bert.transformer_blocks
        target_transformer_blocks = target_model.transformer_blocks

        # Determine total layers and number of shared layers
        total_layers = len(source_transformer_blocks)
        unshared_layers = min(unshared_layers, total_layers)
        shared_layers = total_layers - unshared_layers

        # Apply layer sharing
        for i in range(shared_layers):
            target_transformer_blocks[i] = source_transformer_blocks[i]

        return target_model

    except AttributeError as e:
        raise AttributeError(f"Model structure mismatch: {str(e)}. Please confirm the model has a transformer_blocks attribute") from e
    except Exception as e:
        raise Exception(f"Error while applying parameter sharing: {str(e)}") from e


def freeze_encoder_layers(encoder, freeze_layers):
    """
    Freeze encoder layers; supports different model architectures.

    :param encoder: encoder to freeze
    :param freeze_layers: int freezes the first n layers; float freezes the first percentage of layers
    """
    # Try to determine the model's layer structure
    if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
        # Standard BERT architecture
        layers = encoder.encoder.layer
    elif hasattr(encoder, 'transformer_blocks'):
        # MSBERT architecture
        layers = encoder.transformer_blocks
    else:
        print(f"Warning: cannot determine model layer structure; freeze skipped")
        return

    # Compute number of layers to freeze
    total_layers = len(layers)
    if isinstance(freeze_layers, float):
        num_freeze = int(total_layers * freeze_layers)
    else:
        num_freeze = min(freeze_layers, total_layers)

    print(f"Freezing first {num_freeze} layers out of {total_layers}")

    # Freeze specified layers
    for i in range(num_freeze):
        for param in layers[i].parameters():
            param.requires_grad = False


def unfreeze_encoder_layers(encoder, unfreeze_ratio):
    """
    Automatically detect frozen encoder layers and unfreeze a portion of them.

    :param encoder: previously frozen encoder
    :param unfreeze_ratio: float in [0, 1], ratio of frozen layers to unfreeze
    """
    # Try to determine the model's layer structure
    if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
        layers = encoder.encoder.layer
    elif hasattr(encoder, 'transformer_blocks'):
        layers = encoder.transformer_blocks
    else:
        print("Warning: cannot determine model layer structure; unfreeze skipped")
        return

    total_layers = len(layers)

    # Auto-detect how many leading layers are frozen (i.e., requires_grad=False)
    frozen_layer_indices = []
    for idx, layer in enumerate(layers):
        params = list(layer.parameters())
        if all(not p.requires_grad for p in params):
            frozen_layer_indices.append(idx)
        else:
            break  # once a trainable layer is hit, the rest are assumed unfrozen

    num_frozen = len(frozen_layer_indices)
    if num_frozen == 0:
        print("Note: no frozen layers detected")
        return

    # Compute how many layers to unfreeze based on the ratio
    num_to_unfreeze = max(1, int(num_frozen * unfreeze_ratio))
    print(f"Detected {num_frozen} frozen layers; unfreezing the first {num_to_unfreeze} of them (ratio {unfreeze_ratio})")

    for i in range(num_to_unfreeze):
        for param in layers[frozen_layer_indices[i]].parameters():
            param.requires_grad = True



def freeze_embedding_layers(model, freeze_embedding=True):
    """
    Freeze the model's embedding layer.

    :param model: model whose embeddings should be frozen
    :param freeze_embedding: whether to freeze the embedding layer
    """
    if not freeze_embedding:
        return

    # Try to determine the embedding layer structure
    if hasattr(model, 'embeddings'):
        # Standard BERT architecture
        embeddings = model.embeddings
        print(f"Freezing standard BERT embedding layer")
        for param in embeddings.parameters():
            param.requires_grad = False
    elif hasattr(model, 'embedding') and hasattr(model.embedding, 'token'):
        # MSBERT architecture
        print(f"Freezing MSBERT embedding layer")
        for param in model.embedding.parameters():
            param.requires_grad = False
        # Also freeze MS-BERT's fc2 (output layer); it usually shares size with embeddings
        if hasattr(model, 'fc2'):
            for param in model.fc2.parameters():
                param.requires_grad = False
    else:
        print(f"Warning: cannot determine model embedding structure; freeze skipped")


def get_hidden_size(model, default_size=None):
    """
    Try to obtain the hidden size from the model, including custom models.
    """
    try:
        return model.config.hidden_size
    except AttributeError:
        pass

    try:
        return model.hidden_size  # if you manually added this attribute
    except AttributeError:
        pass

    # Try to infer the hidden size from Linear layers etc.
    try:
        if hasattr(model, "linear") and isinstance(model.linear, nn.Linear):
            return model.linear.in_features
    except:
        pass

    print("[Warning] Could not get hidden_size from the model; using default:", default_size)
    return default_size
