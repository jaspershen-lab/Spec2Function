import copy
import torch.nn as nn

def apply_partial_sharing(ms_bert, unshared_layers):
    """
    共享前 n 层，并让最后 unshared_layers 层独立
    
    参数:
        ms_bert: 原始 MSBERT 编码器
        unshared_layers: 需要独立训练的层数（整数）
        
    返回:
        部分共享参数的编码器
    """
    try:
        # 创建新的编码器
        target_model = copy.deepcopy(ms_bert)  # 先复制整个模型
        
        # 获取源模型和目标模型的transformer块
        source_transformer_blocks = ms_bert.transformer_blocks
        target_transformer_blocks = target_model.transformer_blocks
        
        # 确定总层数和需要共享的层数
        total_layers = len(source_transformer_blocks)
        unshared_layers = min(unshared_layers, total_layers)
        shared_layers = total_layers - unshared_layers
        
        # 实现层共享
        for i in range(shared_layers):
            target_transformer_blocks[i] = source_transformer_blocks[i]
            
        return target_model
        
    except AttributeError as e:
        raise AttributeError(f"模型结构不匹配: {str(e)}。请确认模型有transformer_blocks属性") from e
    except Exception as e:
        raise Exception(f"应用参数共享时发生错误: {str(e)}") from e
    

def freeze_encoder_layers(encoder, freeze_layers):
    """
    冻结encoder的层，适应不同模型架构。
    
    :param encoder: 需要冻结的编码器
    :param freeze_layers: int表示冻结前n层, float表示冻结前freeze_layers百分比的层
    """
    # 尝试确定模型的层结构
    if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
        # 标准BERT架构
        layers = encoder.encoder.layer
    elif hasattr(encoder, 'transformer_blocks'):
        # MSBERT架构
        layers = encoder.transformer_blocks
    else:
        print(f"警告: 无法确定模型的层结构，冻结操作未执行")
        return
    
    # 计算需要冻结的层数
    total_layers = len(layers)
    if isinstance(freeze_layers, float):
        num_freeze = int(total_layers * freeze_layers)
    else:
        num_freeze = min(freeze_layers, total_layers)
    
    print(f"冻结前{num_freeze}层，共{total_layers}层")
    
    # 冻结指定层
    for i in range(num_freeze):
        for param in layers[i].parameters():
            param.requires_grad = False


def unfreeze_encoder_layers(encoder, unfreeze_ratio):
    """
    自动检测被冻结的encoder层，并按比例解冻一部分。
    
    :param encoder: 冻结过的编码器
    :param unfreeze_ratio: float, 表示解冻已冻结层的比例（0~1之间）
    """
    # 尝试确定模型的层结构
    if hasattr(encoder, 'encoder') and hasattr(encoder.encoder, 'layer'):
        layers = encoder.encoder.layer
    elif hasattr(encoder, 'transformer_blocks'):
        layers = encoder.transformer_blocks
    else:
        print("警告: 无法确定模型的层结构，解冻操作未执行")
        return

    total_layers = len(layers)

    # 自动检测前面多少层被冻结（即 requires_grad=False）
    frozen_layer_indices = []
    for idx, layer in enumerate(layers):
        params = list(layer.parameters())
        if all(not p.requires_grad for p in params):
            frozen_layer_indices.append(idx)
        else:
            break  # 一旦遇到可训练层，后面默认都是未冻结的

    num_frozen = len(frozen_layer_indices)
    if num_frozen == 0:
        print("提示: 当前没有检测到被冻结的层")
        return

    # 按照解冻比例计算要解冻的层数
    num_to_unfreeze = max(1, int(num_frozen * unfreeze_ratio))
    print(f"检测到共冻结了{num_frozen}层，将解冻其中前{num_to_unfreeze}层（比例 {unfreeze_ratio}）")

    for i in range(num_to_unfreeze):
        for param in layers[frozen_layer_indices[i]].parameters():
            param.requires_grad = True



def freeze_embedding_layers(model, freeze_embedding=True):
    """
    冻结模型的embedding层
    
    :param model: 需要冻结embedding的模型
    :param freeze_embedding: 是否冻结embedding层
    """
    if not freeze_embedding:
        return
        
    # 尝试确定embedding层的结构
    if hasattr(model, 'embeddings'):
        # 标准BERT架构
        embeddings = model.embeddings
        print(f"冻结standard BERT embedding层")
        for param in embeddings.parameters():
            param.requires_grad = False
    elif hasattr(model, 'embedding') and hasattr(model.embedding, 'token'):
        # MSBERT架构
        print(f"冻结MSBERT embedding层")
        for param in model.embedding.parameters():
            param.requires_grad = False
        # 同时冻结MS-BERT的fc2层(输出层)，它通常与embedding大小相同
        if hasattr(model, 'fc2'):
            for param in model.fc2.parameters():
                param.requires_grad = False
    else:
        print(f"警告: 无法确定模型的embedding结构，冻结操作未执行")


def get_hidden_size(model, default_size=None):
    """
    尝试从模型获取隐藏层维度，包括自定义模型
    """
    try:
        return model.config.hidden_size
    except AttributeError:
        pass

    try:
        return model.hidden_size  # 如果你手动加过这个属性
    except AttributeError:
        pass

    # 尝试从 Linear 层等结构猜出隐藏维度
    try:
        if hasattr(model, "linear") and isinstance(model.linear, nn.Linear):
            return model.linear.in_features
    except:
        pass

    print("[警告] 无法从模型中获取 hidden_size，使用默认值:", default_size)
    return default_size
