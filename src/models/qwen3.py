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
        if tokenizer.pad_token is None:
            tokenizer.pad_token = "<|endoftext|>"
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")


        # Right padding for autoregressive LMs
        tokenizer.padding_side = "right"
        return tokenizer

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            # attn_implementation="eager",  # important to compatibility with ROCm accelerators
            # torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16,
        )
        
        model.config.use_cache = False # Disable cache for PEFT training
        return model.to(self.device)
