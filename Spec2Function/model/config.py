import argparse
# TODO: this should probably be deprecated
def get_base_config():
    """
    Get the base MS2BioText configuration.
    """
    parser = argparse.ArgumentParser(description='MS2BioText CLIP model configuration')

    # ========== Embedding space ==========
    parser.add_argument('--embedding_dim', type=int, default=512,
                        help='shared embedding space dimension')

    # ========== Projection head ==========
    parser.add_argument('--projection_dropout', type=float, default=0.1,
                        help='projection head dropout rate')

    # ========== Text encoding ==========
    parser.add_argument('--text_pooling', type=str, default='cls',
                        choices=['cls', 'mean', 'max'],
                        help='text pooling strategy')

    # ========== Contrastive learning ==========
    parser.add_argument('--temperature', type=float, default=0.07,
                        help='contrastive learning temperature')
    parser.add_argument('--learnable_temperature', type=bool, default=True,
                        help='whether the temperature is learnable')
    parser.add_argument('--symmetric_loss', type=bool, default=True,
                        help='whether to use symmetric contrastive loss')

    # ========== Auxiliary tasks ==========
    parser.add_argument('--use_mlm', type=bool, default=False,
                        help='whether to enable masked language modeling')
    parser.add_argument('--mlm_loss_weight', type=float, default=0.1,
                        help='MLM loss weight')
    parser.add_argument('--use_ms2_prediction', type=bool, default=False,
                        help='whether to enable MS2 prediction task')
    parser.add_argument('--ms2_prediction_loss_weight', type=float, default=0.1,
                        help='MS2 prediction loss weight')

    # ========== Parameter freezing ==========
    # MS2 encoder freezing
    parser.add_argument('--freeze_ms_embedding', type=bool, default=False,
                        help='whether to freeze the MS2 embedding layer')
    parser.add_argument('--freeze_ms_encoder', type=float, default=0,
                        help='freeze MS2 encoder: int = first N layers, float = first N%% of layers, 0 = no freezing')

    # Text encoder freezing
    parser.add_argument('--freeze_text_embedding', type=bool, default=False,
                        help='whether to freeze the text embedding layer')
    parser.add_argument('--freeze_text_encoder', type=float, default=0,
                        help='freeze text encoder: int = first N layers, float = first N%% of layers, 0 = no freezing')

    return parser.parse_args([])

def get_projection_only_config():
    """
    Get the projection-only training configuration (fully freezes pretrained encoders).
    """
    config = get_base_config()

    # Fully freeze all pretrained encoders
    config.freeze_ms_embedding = True
    config.freeze_ms_encoder = 1.0  # freeze 100% of layers
    config.freeze_text_embedding = True
    config.freeze_text_encoder = 1.0  # freeze 100% of layers

    # Smaller embedding dim for fast training
    config.embedding_dim = 256
    config.projection_dropout = 0.2

    # No auxiliary tasks
    config.use_mlm = False
    config.use_ms2_prediction = False

    return config

def get_partial_freeze_config():
    """
    Get the partial-freeze configuration (freezes most encoder layers).
    """
    config = get_base_config()

    # Freeze the first 80% of layers
    config.freeze_ms_encoder = 0.8
    config.freeze_text_encoder = 0.8
    config.freeze_ms_embedding = False  # allow embedding layer to fine-tune
    config.freeze_text_embedding = False

    # Enable the MLM auxiliary task
    config.use_mlm = True
    config.mlm_loss_weight = 0.1

    return config

def get_top_layers_freeze_config():
    """
    Get the top-layer fine-tuning configuration (only the last few layers are unfrozen).
    """
    config = get_base_config()

    # Freeze the first 10 layers (assuming the encoder has 12 layers)
    config.freeze_ms_encoder = 10
    config.freeze_text_encoder = 10
    config.freeze_ms_embedding = False
    config.freeze_text_embedding = False

    # Enable all auxiliary tasks
    config.use_mlm = True
    config.use_ms2_prediction = True
    config.mlm_loss_weight = 0.1
    config.ms2_prediction_loss_weight = 0.1

    return config

def get_full_finetune_config():
    """
    Get the full fine-tuning configuration (all parameters trainable).
    """
    config = get_base_config()

    # Don't freeze any parameter
    config.freeze_ms_encoder = 0
    config.freeze_text_encoder = 0
    config.freeze_ms_embedding = False
    config.freeze_text_embedding = False

    # Enable all auxiliary tasks
    config.use_mlm = True
    config.use_ms2_prediction = True
    config.mlm_loss_weight = 0.1
    config.ms2_prediction_loss_weight = 0.1

    # Larger embedding dimension
    config.embedding_dim = 768

    return config

def get_inference_config():
    """
    Get the inference configuration.
    """
    config = get_base_config()

    # No auxiliary tasks needed at inference time
    config.use_mlm = False
    config.use_ms2_prediction = False

    # Fixed temperature
    config.learnable_temperature = False

    # Efficient pooling strategy
    config.text_pooling = 'cls'

    return config

def get_progressive_training_configs():
    """
    Get a sequence of configurations for progressive training.
    """
    return {
        'stage1_projection_only': get_projection_only_config(),
        'stage2_partial_freeze': get_partial_freeze_config(),
        'stage3_top_layers': get_top_layers_freeze_config(),
        'stage4_full_finetune': get_full_finetune_config()
    }

# ========== Config manager ==========
class ConfigManager:
    """Configuration manager."""

    @staticmethod
    def get_config(config_type='base'):
        """
        Get a configuration by type.

        :param config_type: configuration type
        :return: configuration object
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
            raise ValueError(f"Unsupported config type: {config_type}")

        return config_map[config_type]()

    @staticmethod
    def get_progressive_configs():
        """Get progressive training configurations."""
        return get_progressive_training_configs()

    @staticmethod
    def print_freeze_info(config):
        """Print freezing strategy information."""
        print(f"\n=== Freezing strategy ===")
        print(f"MS2 encoder freezing: {config.freeze_ms_encoder} ({'layers' if isinstance(config.freeze_ms_encoder, int) else 'percentage'})")
        print(f"Text encoder freezing: {config.freeze_text_encoder} ({'layers' if isinstance(config.freeze_text_encoder, int) else 'percentage'})")
        print(f"MS2 embedding frozen: {config.freeze_ms_embedding}")
        print(f"Text embedding frozen: {config.freeze_text_embedding}")

    @staticmethod
    def print_config(config, title="Config info"):
        """Print configuration information."""
        print(f"\n=== {title} ===")
        print(f"Embedding dim: {config.embedding_dim}")
        print(f"Projection dropout: {config.projection_dropout}")
        print(f"Text pooling: {config.text_pooling}")
        print(f"Contrastive temperature: {config.temperature}")
        print(f"Learnable temperature: {config.learnable_temperature}")
        print(f"Symmetric loss: {config.symmetric_loss}")
        print(f"Use MLM: {config.use_mlm}")
        if config.use_mlm:
            print(f"MLM loss weight: {config.mlm_loss_weight}")
        print(f"Use MS2 prediction: {config.use_ms2_prediction}")
        if config.use_ms2_prediction:
            print(f"MS2 prediction loss weight: {config.ms2_prediction_loss_weight}")

        ConfigManager.print_freeze_info(config)

# ========== Usage example ==========
if __name__ == "__main__":
    print("=== MS2BioText config system ===")

    # Demonstrate different freezing strategies
    configs = {
        'Projection only': ConfigManager.get_config('projection_only'),
        'Partial freeze': ConfigManager.get_config('partial_freeze'),
        'Top-layer fine-tune': ConfigManager.get_config('top_layers'),
        'Full fine-tune': ConfigManager.get_config('full_finetune'),
        'Inference mode': ConfigManager.get_config('inference')
    }

    for name, config in configs.items():
        ConfigManager.print_config(config, f"{name} config")
        print("-" * 50)

    # Demonstrate progressive training configs
    print(f"\n=== Progressive training config sequence ===")
    progressive_configs = ConfigManager.get_progressive_configs()

    for stage_name, stage_config in progressive_configs.items():
        print(f"\n{stage_name}:")
        ConfigManager.print_freeze_info(stage_config)
