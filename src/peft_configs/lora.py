from peft import LoraConfig, get_peft_model, TaskType
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoraAdapter():
    def __init__(self, model, lora_preset="default", **additional_configs):
        """Initialize LoRA configuration.
        Args:
            model: The model to apply LoRA to
            lora_preset (str, optional): Preset LoRA configuration name to use. Defaults to None.
            **lora_configs: Custom LoRA configuration parameters if no preset is used
        
        Available presets:
            - default: Basic configuration (r=8, alpha=16)
            - large: Higher rank configuration (r=16, alpha=32)
            - baseline: Minimal q_proj only configuration (r=4)
            - control_plus_content: Query and value projections (r=4)
            - full_attention: All attention components (r=8)
            - ffn: Feed-forward network only (r=4)
            - attention_plus_ffn: Combined attention and FFN (r=4)
            - low_rank: Lower rank configuration (r=4, alpha=8)
            - high_rank: Higher rank configuration (r=32, alpha=64)
        
        The class handles LoRA (Low-Rank Adaptation) configuration either through predefined
        presets or custom parameters. If a preset name is provided, it will use the corresponding
        preset configuration, otherwise it will use the custom parameters passed in lora_configs.
        """
        self.model = model
        self.presets = {
            "default": {
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "large": {
                "r": 16,
                "lora_alpha": 32,
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "baseline": {
                "r": 4,
                "target_modules": ["q_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "control_plus_content": {
                "r": 4,
                "target_modules": ["q_proj", "v_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "full_attention": {##
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "all": {##
                "r": 8,
                "lora_alpha": 16,
                "target_modules": "all-linear",
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "ffn": {##
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["mlp.up_proj", "mlp.down_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "attention_plus_ffn": {
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "v_proj", "mlp.up_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "full_attention_plus_ffn": {##
                "r": 8,
                "lora_alpha": 16,
                "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "mlp.up_proj", "mlp.down_proj"],
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "low_rank": {##
                "r": 4,
                "lora_alpha": 8,
                "target_modules": "all-linear",
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "high_rank": {##
                "r": 32,
                "lora_alpha": 64,
                "target_modules": "all-linear",
                "lora_dropout": 0.1,
                "bias": "none",
            },
            "high_rank_XX": {##
                "r": 128,
                "lora_alpha": 256,
                "target_modules": "all-linear",
                "lora_dropout": 0.1,
                "bias": "none",
            }
        }
        self.lora_preset = lora_preset
        if self.lora_preset is None:
            logging.info("No LoRA preset provided. Using additional configuration provided.")
        elif lora_preset not in self.presets:
            raise ValueError(f"Invalid LoRA preset: {lora_preset}. Available presets: {list(self.presets.keys())}")
        
        self.additional_configs = additional_configs if additional_configs else {}
    
    def apply_lora(self):
        if self.lora_preset:
            self.lora_configs = self.presets[self.lora_preset]
        else:
            self.lora_configs = {}
        self.lora_configs.update({
            "task_type": TaskType.CAUSAL_LM
        })
        self.lora_configs.update(self.additional_configs)
        
        logging.info(f"Applying LoRA with preset: {self.lora_preset}")
        logging.info(f"LoRA configurations: {self.lora_configs}")
        
        peft_config = LoraConfig(**self.lora_configs)
        self.model = get_peft_model(self.model, peft_config)
        self.model.print_trainable_parameters()

        return self.model
