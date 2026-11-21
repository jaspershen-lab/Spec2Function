import argparse
# TODO:这个应该被废弃了
def get_base_config():
    """
    获取MS2BioText基础配置
    """
    parser = argparse.ArgumentParser(description='MS2BioText CLIP模型配置')
    
    # ========== 嵌入空间配置 ==========
    parser.add_argument('--embedding_dim', type=int, default=512, 
                        help='共同嵌入空间维度')
    
    # ========== 投影头配置 ==========
    parser.add_argument('--projection_dropout', type=float, default=0.1, 
                        help='投影头Dropout率')
    
    # ========== 文本编码配置 ==========
    parser.add_argument('--text_pooling', type=str, default='cls', 
                        choices=['cls', 'mean', 'max'],
                        help='文本池化策略')
    
    # ========== 对比学习配置 ==========
    parser.add_argument('--temperature', type=float, default=0.07, 
                        help='对比学习温度参数')
    parser.add_argument('--learnable_temperature', type=bool, default=True, 
                        help='温度参数是否可学习')
    parser.add_argument('--symmetric_loss', type=bool, default=True, 
                        help='是否使用对称对比损失')
    
    # ========== 辅助任务配置 ==========
    parser.add_argument('--use_mlm', type=bool, default=False, 
                        help='是否启用掩码语言建模')
    parser.add_argument('--mlm_loss_weight', type=float, default=0.1, 
                        help='MLM损失权重')
    parser.add_argument('--use_ms2_prediction', type=bool, default=False, 
                        help='是否启用MS2预测任务')
    parser.add_argument('--ms2_prediction_loss_weight', type=float, default=0.1, 
                        help='MS2预测损失权重')
    
    # ========== 参数冻结配置 ==========
    # MS2编码器冻结
    parser.add_argument('--freeze_ms_embedding', type=bool, default=False, 
                        help='是否冻结MS2嵌入层')
    parser.add_argument('--freeze_ms_encoder', type=float, default=0, 
                        help='冻结MS2编码器：int表示前N层，float表示前N%层，0表示不冻结')
    
    # 文本编码器冻结
    parser.add_argument('--freeze_text_embedding', type=bool, default=False, 
                        help='是否冻结文本嵌入层')
    parser.add_argument('--freeze_text_encoder', type=float, default=0, 
                        help='冻结文本编码器：int表示前N层，float表示前N%层，0表示不冻结')
    
    return parser.parse_args([])

def get_projection_only_config():
    """
    获取仅训练投影头配置（完全冻结预训练编码器）
    """
    config = get_base_config()
    
    # 完全冻结所有预训练编码器
    config.freeze_ms_embedding = True
    config.freeze_ms_encoder = 1.0  # 冻结100%的层
    config.freeze_text_embedding = True
    config.freeze_text_encoder = 1.0  # 冻结100%的层
    
    # 较小的嵌入维度用于快速训练
    config.embedding_dim = 256
    config.projection_dropout = 0.2
    
    # 不使用辅助任务
    config.use_mlm = False
    config.use_ms2_prediction = False
    
    return config

def get_partial_freeze_config():
    """
    获取部分冻结配置（冻结大部分编码器层）
    """
    config = get_base_config()
    
    # 冻结前80%的层
    config.freeze_ms_encoder = 0.8
    config.freeze_text_encoder = 0.8
    config.freeze_ms_embedding = False  # 允许嵌入层微调
    config.freeze_text_embedding = False
    
    # 启用MLM辅助任务
    config.use_mlm = True
    config.mlm_loss_weight = 0.1
    
    return config

def get_top_layers_freeze_config():
    """
    获取顶层解冻配置（只解冻最后几层）
    """
    config = get_base_config()
    
    # 冻结前10层（假设编码器有12层）
    config.freeze_ms_encoder = 10
    config.freeze_text_encoder = 10
    config.freeze_ms_embedding = False
    config.freeze_text_embedding = False
    
    # 启用所有辅助任务
    config.use_mlm = True
    config.use_ms2_prediction = True
    config.mlm_loss_weight = 0.1
    config.ms2_prediction_loss_weight = 0.1
    
    return config

def get_full_finetune_config():
    """
    获取全量微调配置（解冻所有参数）
    """
    config = get_base_config()
    
    # 不冻结任何参数
    config.freeze_ms_encoder = 0
    config.freeze_text_encoder = 0
    config.freeze_ms_embedding = False
    config.freeze_text_embedding = False
    
    # 启用所有辅助任务
    config.use_mlm = True
    config.use_ms2_prediction = True
    config.mlm_loss_weight = 0.1
    config.ms2_prediction_loss_weight = 0.1
    
    # 更大的嵌入维度
    config.embedding_dim = 768
    
    return config

def get_inference_config():
    """
    获取推理配置
    """
    config = get_base_config()
    
    # 推理时不需要辅助任务
    config.use_mlm = False
    config.use_ms2_prediction = False
    
    # 固定温度参数
    config.learnable_temperature = False
    
    # 高效池化策略
    config.text_pooling = 'cls'
    
    return config

def get_progressive_training_configs():
    """
    获取渐进式训练的配置序列
    """
    return {
        'stage1_projection_only': get_projection_only_config(),
        'stage2_partial_freeze': get_partial_freeze_config(), 
        'stage3_top_layers': get_top_layers_freeze_config(),
        'stage4_full_finetune': get_full_finetune_config()
    }

# ========== 配置管理器 ==========
class ConfigManager:
    """配置管理器"""
    
    @staticmethod
    def get_config(config_type='base'):
        """
        根据类型获取配置
        
        :param config_type: 配置类型
        :return: 配置对象
        """
        config_map = {
            'base': get_base_config,
            'projection_only': get_projection_only_config,
            'partial_freeze': get_partial_freeze_config,
            'top_layers': get_top_layers_freeze_config,
            'full_finetune': get_full_finetune_config,
            'inference': get_inference_config
        }
        
        if config_type not in config_map:
            raise ValueError(f"不支持的配置类型: {config_type}")
        
        return config_map[config_type]()
    
    @staticmethod
    def get_progressive_configs():
        """获取渐进式训练配置"""
        return get_progressive_training_configs()
    
    @staticmethod
    def print_freeze_info(config):
        """打印冻结策略信息"""
        print(f"\n=== 冻结策略 ===")
        print(f"MS2编码器冻结: {config.freeze_ms_encoder} ({'层数' if isinstance(config.freeze_ms_encoder, int) else '百分比'})")
        print(f"文本编码器冻结: {config.freeze_text_encoder} ({'层数' if isinstance(config.freeze_text_encoder, int) else '百分比'})")
        print(f"MS2嵌入层冻结: {config.freeze_ms_embedding}")
        print(f"文本嵌入层冻结: {config.freeze_text_embedding}")
    
    @staticmethod
    def print_config(config, title="配置信息"):
        """打印配置信息"""
        print(f"\n=== {title} ===")
        print(f"嵌入维度: {config.embedding_dim}")
        print(f"投影头Dropout: {config.projection_dropout}")
        print(f"文本池化策略: {config.text_pooling}")
        print(f"对比学习温度: {config.temperature}")
        print(f"可学习温度: {config.learnable_temperature}")
        print(f"对称损失: {config.symmetric_loss}")
        print(f"使用MLM: {config.use_mlm}")
        if config.use_mlm:
            print(f"MLM损失权重: {config.mlm_loss_weight}")
        print(f"使用MS2预测: {config.use_ms2_prediction}")
        if config.use_ms2_prediction:
            print(f"MS2预测损失权重: {config.ms2_prediction_loss_weight}")
        
        ConfigManager.print_freeze_info(config)

# ========== 使用示例 ==========
if __name__ == "__main__":
    print("=== MS2BioText配置系统 ===")
    
    # 演示不同的冻结策略
    configs = {
        '仅投影头': ConfigManager.get_config('projection_only'),
        '部分冻结': ConfigManager.get_config('partial_freeze'),
        '顶层微调': ConfigManager.get_config('top_layers'),
        '全量微调': ConfigManager.get_config('full_finetune'),
        '推理模式': ConfigManager.get_config('inference')
    }
    
    for name, config in configs.items():
        ConfigManager.print_config(config, f"{name}配置")
        print("-" * 50)
    
    # 演示渐进式训练配置
    print(f"\n=== 渐进式训练配置序列 ===")
    progressive_configs = ConfigManager.get_progressive_configs()
    
    for stage_name, stage_config in progressive_configs.items():
        print(f"\n{stage_name}:")
        ConfigManager.print_freeze_info(stage_config)