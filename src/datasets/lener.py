import logging
from datasets import load_dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LenerDataset:
    def __init__(self, dataset_name: str = "eduagarcia/PortuLex_benchmark", tokenizer=None):
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset: DatasetDict | None = None
        self.tag_id_to_name = {}
        self.name_to_tag_id = {}

    def load_dataset(self):
        logger.info(f"Loading dataset: {self.dataset_name}")
        # loaded_data = load_dataset(self.dataset_name, trust_remote_code=True)
        loaded_data = load_dataset(self.dataset_name,'LeNER-Br')
        if not isinstance(loaded_data, DatasetDict):
            raise TypeError(f"Expected load_dataset to return a DatasetDict, but got {type(loaded_data)}")
        self.dataset = loaded_data
        logger.info(f"Dataset loaded with splits: {list(self.dataset.keys())}")
        self.dataset = self.format_dataset_GRPO()
        return self.dataset

    def format_dataset(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        if not isinstance(self.dataset, DatasetDict):
            raise TypeError(f"self.dataset is not a DatasetDict (type: {type(self.dataset)}). Cannot proceed.")
        if not self.dataset:
            raise ValueError("DatasetDict is empty. No splits found.")

        split_names = list(self.dataset.keys())
        logger.info(f"Processing splits: {split_names}")
        first_split_key = split_names[0]

        original_columns = self.dataset[first_split_key].column_names
        logger.info(f"Original columns to remove after mapping: {original_columns}")

        try:
            ner_feature = self.dataset[first_split_key].features["ner_tags"]
            self.tag_id_to_name = {i: name for i, name in enumerate(ner_feature.feature.names)}
            logger.info(f"NER tag mapping created: {self.tag_id_to_name}")
        except (KeyError, AttributeError) as e:
            logger.error(f"Failed to get NER tag feature names from dataset split '{first_split_key}'. Error: {e}")
            raise ValueError("Could not extract NER tag information. Check dataset structure.") from e

        logger.info("Starting dataset mapping...")
        formatted_dataset = self.dataset.map(
            self.format_example,
            batched=False,
            load_from_cache_file=False,
            # remove_columns=original_columns
        )
        logger.info("Dataset mapping finished.")
        logger.info(f"Columns after mapping: {formatted_dataset[first_split_key].column_names}")

        return formatted_dataset

    def format_example(self, example):
        if "ner_tags" not in example or "tokens" not in example:
            logger.warning(f"Skipping example due to missing keys: {example.keys()}")
            return {}
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
        )
        sentence = " ".join(example["tokens"])
        entities = self.extract_entities(example["tokens"], example["ner_tags"])
        input_text = f"{context_prompt}Texto: {sentence}"
        target_text = "Resposta:\n " + "; ".join(entities) if entities else "Resposta:\n Nenhuma"
        full_text = input_text + "\n" + target_text + self.tokenizer.eos_token

        try:
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=1024
            )
            labels = tokenized["input_ids"].copy()
            
            # Find the last occurrence of the EOS token
            eos_indices = [i for i, token_id in enumerate(labels) if token_id == self.tokenizer.eos_token_id]
            if eos_indices:
                last_eos_idx = eos_indices[-1]
                # Mask only the padding tokens that appear *after* the last EOS token
                for i in range(last_eos_idx + 1, len(labels)):
                    # if labels[i] == self.tokenizer.pad_token_id:
                    labels[i] = -100

            tokenized["labels"] = labels
        except Exception as e:
            logger.error(f"Tokenization failed for text: '{full_text[:100]}...'. Error: {e}")
            return {}

        return tokenized

    def extract_entities(self, tokens, tags):
        entities = []
        current_ent = ""
        current_tag = None

        for token, tag_id in zip(tokens, tags):
            tag_name = self.tag_id_to_name.get(tag_id, "O")

            if tag_name.startswith("B-"):
                if current_ent:
                    entities.append(f"{current_tag}: {current_ent.strip()}")
                current_tag = tag_name[2:]
                current_ent = token + " "
            elif tag_name.startswith("I-") and current_tag:
                if tag_name[2:] == current_tag:
                    current_ent += token + " "
                else:
                    entities.append(f"{current_tag}: {current_ent.strip()}")
                    current_tag = tag_name[2:]
                    current_ent = token + " "
            else:
                if current_ent:
                    entities.append(f"{current_tag}: {current_ent.strip()}")
                    current_ent = ""
                    current_tag = None

        if current_ent:
            entities.append(f"{current_tag}: {current_ent.strip()}")

        return entities

    def format_example_IOB(self, example):
        if "ner_tags" not in example or "tokens" not in example:
            logger.warning(f"Skipping example due to missing keys: {example.keys()}")
            return {}
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
        )
        sentence = " ".join(example["tokens"])
        input_text = f"{context_prompt}Texto: {sentence}"
        target_text = "Resposta:\n"
        for token, tag_id in zip(example["tokens"], example["ner_tags"]):
            tag_name = self.tag_id_to_name.get(tag_id, "O")
            target_text += f"{token}:{tag_name}\n"
        full_text = input_text + "\n" + target_text + self.tokenizer.eos_token
        # logger.warning(f"fulltext: {full_text}")
        # logger.info("Starting dataset mapping...")
        try:
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=1536
            )
            labels = tokenized["input_ids"].copy()
            
            eos_indices = [i for i, token_id in enumerate(labels) if token_id == self.tokenizer.eos_token_id]
            if eos_indices:
                last_eos_idx = eos_indices[-1]
                # Mask only the padding tokens that appear *after* the last EOS token
                for i in range(last_eos_idx + 1, len(labels)):
                    # if labels[i] == self.tokenizer.pad_token_id:
                    labels[i] = -100
            
            tokenized["labels"] = labels

            prompt = tokenized["input_ids"].copy()
            resposta_start_idx = self.find_resposta_start(prompt)
            cropped_input_ids = self.build_final_input(prompt, resposta_start_idx)
            tokenized["prompt"] = cropped_input_ids
        except Exception as e:
            logger.error(f"Tokenization failed for text: '{full_text[:100]}...'. Error: {e}")
            return {}

        return tokenized

    def format_dataset_IOB(self):
        if self.dataset is None:
            raise ValueError("Dataset not loaded. Call load_dataset() first.")
        if not isinstance(self.dataset, DatasetDict):
            raise TypeError(f"self.dataset is not a DatasetDict (type: {type(self.dataset)}). Cannot proceed.")
        if not self.dataset:
            raise ValueError("DatasetDict is empty. No splits found.")

        split_names = list(self.dataset.keys())
        logger.info(f"Processing splits: {split_names}")
        first_split_key = split_names[0]

        original_columns = self.dataset[first_split_key].column_names
        logger.info(f"Original columns to remove after mapping: {original_columns}")

        try:
            ner_feature = self.dataset[first_split_key].features["ner_tags"]
            self.tag_id_to_name = {i: name for i, name in enumerate(ner_feature.feature.names)}
            self.name_to_tag_id = {name: i for i, name in enumerate(ner_feature.feature.names)}
            logger.info(f"NER tag mapping created: {self.tag_id_to_name}")
        except (KeyError, AttributeError) as e:
            logger.error(f"Failed to get NER tag feature names from dataset split '{first_split_key}'. Error: {e}")
            raise ValueError("Could not extract NER tag information. Check dataset structure.") from e

        logger.info("Starting dataset mapping...")
        formatted_dataset = self.dataset.map(
            self.format_example_IOB,
            batched=False,
            load_from_cache_file=False,
            # remove_columns=original_columns
        )
        logger.info("Dataset mapping finished.")
        logger.info(f"Columns after mapping: {formatted_dataset[first_split_key].column_names}")

        return formatted_dataset

    def build_final_input(self, original_inputs, resposta_start_idx):
        """
        Faz o corte dos input_ids a partir do índice de início da resposta.
        
        original_inputs: data[idx]["input_ids"]
        resposta_start_idx: saída da função find_resposta_start
        """
        # Cortamos os input_ids a partir do índice encontrado
        cropped_input_ids = original_inputs[:resposta_start_idx]
        
        return cropped_input_ids

    def find_resposta_start(self, input_ids):
        """
        Encontra o índice do token logo após "Resposta:\n" na sequência de input_ids.
        
        input_ids: Lista de IDs de tokens, data[idx]["input_ids"]
        """
        try:
            # retorna o índice do token após "Resposta:\n"
            for idx, _ in enumerate(input_ids):
                if input_ids[idx] == 1061 and input_ids[idx+1] == 38531 and input_ids[idx+2] == 510:
                    return idx + 3
        except:
            # Resposta:\n não encontrado, retornamos última posição de token vista
            return idx
    
    def format_dataset_GRPO(self):
        """
        Formats dataset specifically for TRL's GRPOTrainer.
        Returns columns: ['prompt', 'ground_truth']
        """
        if self.dataset is None:
            self.load_dataset()

        def format_row_grpo(example):
            # 1. Build Prompt (Same as before, but stop before the answer)
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
        )
            sentence = " ".join(example["tokens"])
            input_text = f"{context_prompt}Texto: {sentence}\nResposta:\n"
            
            # 2. Build Ground Truth String (for the reward function to parse)
            # using your existing logic
            entities = self.extract_entities(example["tokens"], example["ner_tags"])
            target_text = "; ".join(entities) if entities else "Nenhuma"

            return {
                "prompt": input_text,
                "ground_truth": target_text # We pass this to the reward function later
            }

        first_split_key = list(self.dataset.keys())[0]
        
        # Get NER feature mappings first
        ner_feature = self.dataset[first_split_key].features["ner_tags"]
        self.tag_id_to_name = {i: name for i, name in enumerate(ner_feature.feature.names)}

        formatted_dataset = self.dataset.map(format_row_grpo, batched=False)
        return formatted_dataset