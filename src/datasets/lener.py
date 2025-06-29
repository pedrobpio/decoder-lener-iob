import logging
from datasets import load_dataset, DatasetDict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LenerDataset:
    def __init__(self, dataset_name: str = "peluz/lener_br", tokenizer=None):
        if tokenizer is None:
            raise ValueError("Tokenizer must be provided.")
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.dataset: DatasetDict | None = None
        self.tag_id_to_name = {}

    def load_dataset(self):
        logger.info(f"Loading dataset: {self.dataset_name}")
        loaded_data = load_dataset(self.dataset_name, trust_remote_code=True)
        if not isinstance(loaded_data, DatasetDict):
            raise TypeError(f"Expected load_dataset to return a DatasetDict, but got {type(loaded_data)}")
        self.dataset = loaded_data
        logger.info(f"Dataset loaded with splits: {list(self.dataset.keys())}")
        self.dataset = self.format_dataset()
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
        target_text = "Entidades: " + "; ".join(entities) if entities else "Entidades: Nenhuma"
        full_text = input_text + "\n" + target_text + self.tokenizer.eos_token

        try:
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                padding="max_length",
                max_length=512
            )
            labels = tokenized["input_ids"].copy()
            labels = [token if token != self.tokenizer.pad_token_id else -100 for token in labels]
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
