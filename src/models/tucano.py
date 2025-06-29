from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Tucano:
    def __init__(self, model_name: str = "TucanoBR/Tucano-1b1", device: str = None):
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
        tokenizer.pad_token = tokenizer.eos_token

        # Right padding for autoregressive LMs
        tokenizer.padding_side = "right"
        return tokenizer

    def load_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            attn_implementation="eager",  # important to compatibility with ROCm accelerators
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        
        model.config.use_cache = False # Disable cache for PEFT training
        return model.to(self.device)
