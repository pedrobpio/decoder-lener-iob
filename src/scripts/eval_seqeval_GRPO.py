"""
Evaluation script for models trained in entity-extraction (GRPO) format.

The model is expected to output lines like:
    PESSOA: João da Silva; ORGANIZACAO: STF; LEGISLACAO: art. 5 CF

Supports both full model checkpoints and LoRA adapter checkpoints.
Uses vLLM for fast batched inference.

Usage:
    python src/scripts/eval_seqeval_GRPO.py \\
        <checkpoint_path> \\
        <base_model_name> \\
        <gpu_id> \\
        [--dataset lener|ulysses] \\
        [--split validation|test] \\
        [--output_dir ./outputs/reports] \\
        [--hf_home /path/to/hf_cache]

Examples:
    # Evaluate GRPO LoRA checkpoint on LeNER-BR validation set
    python src/scripts/eval_seqeval_GRPO.py \\
        grpo/Qwen3-8B_high_rank_64_grpo/checkpoint-150 \\
        Qwen/Qwen3-8B \\
        0 \\
        --dataset lener \\
        --split validation

    # Evaluate full model checkpoint on test set
    python src/scripts/eval_seqeval_GRPO.py \\
        lener_br/Qwen3-8B_high_rank_64/checkpoint-2450 \\
        Qwen/Qwen3-8B \\
        0 \\
        --dataset lener \\
        --split test
"""

import sys
import os
import re
import json
import string
import argparse
from pathlib import Path

import torch
import nltk
from nltk.tokenize import word_tokenize
from seqeval.metrics import classification_report, f1_score
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

nltk.download('punkt_tab', quiet=True)

# ---------------------------------------------------------------------------
# Tag definitions
# ---------------------------------------------------------------------------
LENER_TAGS = ["ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL", "LEGISLACAO", "JURISPRUDENCIA"]
ULYSSES_TAGS = ["ORGANIZACAO", "PESSOA", "DATA", "LOCAL", "EVENTO", "FUNDAMENTO", "PRODUTODELEI"]

LENER_ALLOWED_BIO = [
    "O",
    "B-PESSOA", "I-PESSOA",
    "B-ORGANIZACAO", "I-ORGANIZACAO",
    "B-LOCAL", "I-LOCAL",
    "B-TEMPO", "I-TEMPO",
    "B-LEGISLACAO", "I-LEGISLACAO",
    "B-JURISPRUDENCIA", "I-JURISPRUDENCIA",
]

ULYSSES_ALLOWED_BIO = [
    "O",
    "B-DATA", "I-DATA",
    "B-EVENTO", "I-EVENTO",
    "B-FUNDAMENTO", "I-FUNDAMENTO",
    "B-LOCAL", "I-LOCAL",
    "B-ORGANIZACAO", "I-ORGANIZACAO",
    "B-PESSOA", "I-PESSOA",
    "B-PRODUTODELEI", "I-PRODUTODELEI",
]

LENER_CONTEXT = """Você é um especialista jurídico responsável por identificar entidades em textos.        
As entidades que você deve identificar são:

- ORGANIZACAO: Refere-se a entidades que representam organizações, como empresas, instituições governamentais, ONGs, etc.
- PESSOA: Designa entidades que são nomes de pessoas físicas.
- TEMPO: Marca entidades que expressam informações temporais, como datas, horários, períodos, etc.
- LOCAL: Indica entidades que representam lugares geográficos, como cidades, países, estados, endereços, etc.
- LEGISLACAO: Identifica entidades que correspondem a Atos de Lei, como leis, decretos, portarias, etc.
- JURISPRUDENCIA: Assinala entidades que se referem a decisões relativas a casos legais.      

segue o texto\n"""

ULYSSES_CONTEXT = """Você é um especialista jurídico responsável por identificar entidades em textos.        
As entidades que você deve identificar são:

- ORGANIZACAO: Identifica instituições, entidades administrativas ou grupos formais, como Órgãos do governo, empresas, ONGs e partidos políticos
- PESSOA: Refere-se a seres humanos mencionados nos documentos.
- DATA: Refere-se a marcações temporais presentes nos textos legislativos. como datas, horários, períodos, etc.
- LOCAL: Indica localizações geográficas ou espaços onde ocorrem os trâmites.Incluindo entidades fisicas como países, cidades, prédios e também espaços virtuais como sites oficiais
- EVENTO: Marca acontecimentos específicos que têm nome e duração definida. como Sessões plenárias, conferências, audiências públicas,
- FUNDAMENTO: Refere-se à base legal que sustenta o documento.  Incluindo Leis vigentes, decretos, a Constituição,etc
- PRODUTODELEI: Identifica os resultados práticos ou sistemas gerados a partir de ações legislativas ou governamentais.   Incluindo Softwares, programas, planos de governo, etc

segue o texto\n"""


# ---------------------------------------------------------------------------
# Output parsing: entity-extraction → list of (LABEL, span) tuples
# ---------------------------------------------------------------------------
def parse_grpo_output(text, valid_tags):
    """
    Parses model output in entity-extraction format.
    Strips the 'Resposta:' header if present.
    Returns a list of (TAG, entity_span) tuples (uppercased, stripped).
    """
    if "Resposta:" in text:
        text = text.split("Resposta:")[-1]

    pattern = r"(" + "|".join(valid_tags) + r"):\s*([^;\n]+)"
    matches = re.findall(pattern, text, re.IGNORECASE)
    return [(tag.strip().upper(), span.strip()) for tag, span in matches]


# ---------------------------------------------------------------------------
# Entity → BIO alignment helpers (ported from eval_seqeval3.py)
# ---------------------------------------------------------------------------
def tokenize_text(text):
    return word_tokenize(text)


def find_prefix_before_match(text, known_labels):
    """Truncates entity span text at the first occurrence of any label prefix."""
    if not text:
        return text
    match_patterns = set()
    for label in known_labels:
        if len(label) >= 3:
            for i in range(3, len(label) + 1):
                match_patterns.add(label[:i])
    first_match_index = sys.maxsize
    for pattern in match_patterns:
        index = text.find(pattern)
        if index != -1 and index < first_match_index:
            first_match_index = index
    return text[:first_match_index] if first_match_index != sys.maxsize else text


def clean_entity_text(text):
    return text.rstrip(string.punctuation + string.whitespace)


def filter_and_merge_entities(entities):
    """Deduplicates and removes entities whose span is contained within another same-label entity."""
    if not entities:
        return []
    preprocessed = []
    for label, text in entities:
        cleaned = clean_entity_text(text.split('\n', 1)[0])
        preprocessed.append((label, cleaned))
    unique = list(set(preprocessed))
    if len(unique) <= 1:
        return sorted(unique)
    final = []
    for i, (label_a, text_a) in enumerate(unique):
        subsumed = any(
            label_a == label_b and text_a != text_b and text_a in text_b
            for j, (label_b, text_b) in enumerate(unique) if i != j
        )
        if not subsumed:
            final.append((label_a, text_a))
    return sorted(final)


def match_entities_to_tokens(tokens, filtered_entities, known_labels):
    """Assigns BIO tags to the token list based on matched entity spans."""
    if not tokens:
        return []
    bio_tags = ["O"] * len(tokens)
    num_tokens = len(tokens)

    tokenized_entities = []
    for label, ent_text in filtered_entities:
        trimmed = find_prefix_before_match(ent_text, known_labels)
        cleaned = clean_entity_text(trimmed)
        ent_tokens = tokenize_text(cleaned)
        if ent_tokens:
            tokenized_entities.append({
                "label": label,
                "tokens": ent_tokens,
            })

    tokenized_entities.sort(key=lambda x: len(x["tokens"]), reverse=True)

    for entity_data in tokenized_entities:
        label = entity_data["label"]
        ent_tokens = entity_data["tokens"]
        n = len(ent_tokens)
        if n == 0:
            continue
        for i in range(num_tokens - n + 1):
            if tokens[i: i + n] == ent_tokens:
                if all(bio_tags[i + j] == "O" for j in range(n)):
                    bio_tags[i] = f"B-{label}"
                    for j in range(1, n):
                        bio_tags[i + j] = f"I-{label}"
    return bio_tags


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------
def build_prompt(tokens, context):
    sentence = " ".join(tokens)
    return f"{context}Texto: {sentence}\nResposta:\n"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a GRPO (entity-extraction format) model with vLLM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "checkpoint_path",
        help=(
            "Checkpoint path relative to ./outputs/checkpoints/. "
            "Can be a LoRA adapter directory or a full model directory."
        ),
    )
    parser.add_argument("base_model_name", help="HuggingFace model ID (e.g. Qwen/Qwen3-8B).")
    parser.add_argument("gpu_id", help="GPU index to use (e.g. 0).")
    parser.add_argument(
        "--dataset",
        choices=["lener", "ulysses"],
        default="lener",
        help="Dataset to evaluate on.",
    )
    parser.add_argument(
        "--split",
        choices=["validation", "test"],
        default="validation",
        help="Dataset split to evaluate.",
    )
    parser.add_argument(
        "--output_dir",
        default="./outputs/reports",
        help="Directory where the report .txt file will be saved.",
    )
    parser.add_argument(
        "--hf_home",
        default=None,
        help="Optional path to override HF_HOME (useful on shared clusters).",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=256,
        help="Max new tokens to generate per prompt.",
    )

    args = parser.parse_args()

    if args.hf_home:
        os.environ["HF_HOME"] = args.hf_home

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

    # ------------------------------------------------------------------
    # Resolve checkpoint path and detect LoRA
    # ------------------------------------------------------------------
    checkpoint_abs = str(Path(f"./outputs/checkpoints/{args.checkpoint_path}").resolve())
    is_lora = os.path.exists(os.path.join(checkpoint_abs, "adapter_config.json"))

    if is_lora:
        print(f"Detected LoRA adapter checkpoint: {checkpoint_abs}")
        with open(os.path.join(checkpoint_abs, "adapter_config.json")) as f:
            adapter_conf = json.load(f)
        base_model_path = adapter_conf.get("base_model_name_or_path", args.base_model_name)
        print(f"  Base model: {base_model_path}")
    else:
        base_model_path = checkpoint_abs
        print(f"Full model checkpoint: {checkpoint_abs}")

    tokenizer_path = base_model_path

    # ------------------------------------------------------------------
    # Dataset and tag config
    # ------------------------------------------------------------------
    if args.dataset == "lener":
        from src.datasets.lener import LenerDataset
        valid_tags = LENER_TAGS
        allowed_bio = LENER_ALLOWED_BIO
        context = LENER_CONTEXT
        DatasetClass = LenerDataset
    else:
        from src.datasets.ulysses import UlyssesDataset
        valid_tags = ULYSSES_TAGS
        allowed_bio = ULYSSES_ALLOWED_BIO
        context = ULYSSES_CONTEXT
        DatasetClass = UlyssesDataset

    # ------------------------------------------------------------------
    # Tokenizer and dataset
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    l = DatasetClass(tokenizer=tokenizer)
    data = l.load_dataset(format="raw")

    split_data = data[args.split]
    print(f"Loaded {args.dataset}/{args.split}: {len(split_data)} examples")

    # ------------------------------------------------------------------
    # Build raw text prompts (vLLM takes strings, not token IDs)
    # ------------------------------------------------------------------
    prompts = [
        build_prompt(split_data[i]["tokens"], context)
        for i in range(len(split_data))
    ]

    # ------------------------------------------------------------------
    # vLLM initialization
    # ------------------------------------------------------------------
    print("Initializing vLLM...")
    llm = LLM(
        model=base_model_path,
        tokenizer=tokenizer_path,
        dtype="auto",
        gpu_memory_utilization=0.90,
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_lora=is_lora,
        max_lora_rank=64 if is_lora else 16,
    )

    sampling_params = SamplingParams(
        seed=42,
        temperature=0.0,
        max_tokens=args.max_tokens,
        repetition_penalty=1.0,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else [],
    )

    # ------------------------------------------------------------------
    # Generate
    # ------------------------------------------------------------------
    print(f"Generating for {len(prompts)} prompts...")
    if is_lora:
        outputs = llm.generate(
            prompts=prompts,
            sampling_params=sampling_params,
            lora_request=LoRARequest("ner_adapter", 1, checkpoint_abs),
        )
    else:
        outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)

    decoded_texts = [out.outputs[0].text for out in outputs]
    print(f"Generated {len(decoded_texts)} sequences.")

    # ------------------------------------------------------------------
    # Post-process: entity-extraction → BIO tags
    # ------------------------------------------------------------------
    ner_feature = data["train"].features["ner_tags"]
    tag_id_to_name = {i: name for i, name in enumerate(ner_feature.feature.names)}

    y_true = []
    y_pred = []

    for i, generated_text in enumerate(decoded_texts):
        tokens = split_data[i]["tokens"]
        true_tag_ids = split_data[i]["ner_tags"]
        gold_tags = [tag_id_to_name[t] for t in true_tag_ids]

        raw_entities = parse_grpo_output(generated_text, valid_tags)
        filtered_entities = filter_and_merge_entities(raw_entities)
        pred_tags = match_entities_to_tokens(tokens, filtered_entities, valid_tags)

        # Clamp any hallucinated tags to O
        pred_tags = ["O" if tag not in allowed_bio else tag for tag in pred_tags]

        y_true.append(gold_tags)
        y_pred.append(pred_tags)

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    report = classification_report(y_true, y_pred)
    score = f1_score(y_true, y_pred)
    print("\nClassification Report:\n")
    print(report)
    print(f"F1 Score: {score:.4f}")

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    safe_path = args.checkpoint_path.replace("/", "_")
    output_filename = Path(args.output_dir) / f"{safe_path}_{args.split}_grpo.txt"
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.checkpoint_path}\n")
        f.write(f"Dataset: {args.dataset} / {args.split}\n")
        f.write(f"Format: entity-extraction (GRPO)\n")
        f.write("=" * 50 + "\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nF1 Score: {score:.4f}\n")

    print(f"\nReport saved to: {output_filename}")


if __name__ == "__main__":
    main()
