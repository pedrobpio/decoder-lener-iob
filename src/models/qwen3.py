from transformers import set_seed, AutoModelForCausalLM, AutoTokenizer
import torch
import logging

set_seed(42)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Qwen3:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B", device: str = None):
        self.model_name = model_name
        self.device = device or (
            "mps" if torch.backends.mps.is_available() else
            "cuda" if torch.cuda.is_available() else
            "cpu"
        )
        logging.info(f"Using device: {self.device}")
        self.tokenizer = self.load_tokenizer()
        self.model = self.load_model()

    def load_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Always set pad explicitly to <|endoftext|> (ID 151643) regardless of
        # what the hub config ships with. For Qwen3, EOS is <|im_end|> (151645)
        # — keeping them distinct is critical: pad==eos causes the model to treat
        # padding as a stop signal during generation and corrupts loss masking.
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

        assert tokenizer.pad_token_id != tokenizer.eos_token_id, (
            f"pad_token_id ({tokenizer.pad_token_id}) == eos_token_id "
            f"({tokenizer.eos_token_id}). Fix this or generation will break."
        )

        tokenizer.padding_side = "right"
        return tokenizer

    def load_model(self, adapter_path: str = None):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation="eager",  # required for ROCm compatibility
            dtype=torch.bfloat16 if self.device != "cpu" else torch.float32,
        )

        model.config.use_cache = False
        # Keep model config in sync with tokenizer so Trainer/GRPOTrainer mask
        # the correct token IDs in the loss computation.
        model.config.pad_token_id = self.tokenizer.pad_token_id

        if adapter_path:
            from peft import PeftModel
            logger.info(f"Loading LoRA adapter from: {adapter_path}")
            model = PeftModel.from_pretrained(model, adapter_path)

        return model.to(self.device)
