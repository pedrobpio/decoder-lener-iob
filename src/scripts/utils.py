from transformers import LogitsProcessor, PreTrainedTokenizer, PreTrainedModel
import torch
from typing import List, Set

class RestrictedVocabLogitsProcessor(LogitsProcessor):
    def __init__(self, allowed_token_ids: List[Set[int]]):
        self.allowed_token_ids = allowed_token_ids

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        batch_size = scores.shape[0]
        masked_scores = torch.full_like(scores, float('-inf'))

        for i in range(batch_size):
            ids = self.allowed_token_ids[i]
            masked_scores[i, list(ids)] = scores[i, list(ids)]
        return masked_scores

def get_allowed_token_ids(text: str, tokenizer, extra_tokens=None) -> Set[int]:
    input_token_ids = set(tokenizer(text, add_special_tokens=True)["input_ids"])
    extra_token_ids = set()
    if extra_tokens:
        for token in extra_tokens:
            extra_token_ids.update(tokenizer(token, add_special_tokens=True)["input_ids"])
    return input_token_ids.union(extra_token_ids)

def predict_entities_batch(
    texts: List[str],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    extra_tokens: List[str] = ["PESSOA", "ORGANIZACAO", "LOCAL", "TEMPO", "LEGISLACAO", "JURISPRUDENCIA", ":", ";", "\n", '-', '<|im_end|>']
) -> List[str]:
    
    device = model.device
    context_prompt = (
        """Você é um especialista jurídico responsável por identificar entidades em textos.        
As entidades que você deve identificar são:

- ORGANIZAÇÃO: Refere-se a entidades que representam organizações, como empresas, instituições governamentais, ONGs, etc.
- PESSOA: Designa entidades que são nomes de pessoas físicas.
- TEMPO: Marca entidades que expressam informações temporais, como datas, horários, períodos, etc.
- LOCAL: Indica entidades que representam lugares geográficos, como cidades, países, estados, endereços, etc.
- LEGISLAÇÃO: Identifica entidades que correspondem a Atos de Lei, como leis, decretos, portarias, etc.
- JURISPRUDÊNCIA: Assinala entidades que se referem a decisões relativas a casos legais.      

segue o texto\n"""
#             """Você é um especialista jurídico responsável por identificar entidades de JURISPRUDENCIA em textos.        
# As entidades de JURISPRUDENCIA se referem a decisões relativas a casos legais anteriores.
# segue o texto\n"""
        )
    prompts = [f"{context_prompt}{text}\nResposta:\n" for text in texts]
    tokenized = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
    input_ids = tokenized["input_ids"]

    # at most a token and its corresponding entity
    max_new_tokens = int(input_ids.shape[1] * 2)

    allowed_token_ids_batch = [
        get_allowed_token_ids(text, tokenizer, extra_tokens)
        for text in texts
    ]

    logits_processor = RestrictedVocabLogitsProcessor(allowed_token_ids_batch)

    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        logits_processor=[logits_processor],
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id
    )

    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    return decoded_outputs
