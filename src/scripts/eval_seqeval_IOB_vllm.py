"""
Evaluation script for models trained in IOB (token:TAG per line) format.

Uses vLLM for fast batched inference. Supports LoRA adapter checkpoints.

Usage:
    python src/scripts/eval_seqeval_IOB_vllm.py \\
        <checkpoint_path> \\
        <base_model_name> \\
        <gpu_id> \\
        [--dataset lener|ulysses] \\
        [--split validation|test] \\
        [--output_dir ./outputs/reports] \\
        [--hf_home /path/to/hf_cache]

Examples:
    # LeNER-BR validation set, LoRA checkpoint
    python src/scripts/eval_seqeval_IOB_vllm.py \\
        lener_br/Qwen3-8B_high_rank_64/checkpoint-2450 \\
        Qwen/Qwen3-8B \\
        0 \\
        --dataset lener \\
        --split validation

    # UlyssesNER test set
    python src/scripts/eval_seqeval_IOB_vllm.py \\
        ulysses_v1/Qwen3-14B_high_rank_64/checkpoint-710 \\
        Qwen/Qwen3-14B \\
        0 \\
        --dataset ulysses \\
        --split test
"""

import sys
import os
import json
import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer
from seqeval.metrics import classification_report, f1_score
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

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


def parse_llm_output(llm_text):
    """Parses IOB lines 'token:TAG' into a flat list of tag strings."""
    pred_tags = []
    if "Resposta:\n" in llm_text:
        llm_text = llm_text.split("Resposta:\n")[-1]
    for line in llm_text.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            tag = line.rsplit(":", 1)[-1].strip()
            pred_tags.append(tag)
        else:
            pred_tags.append("O")
    return pred_tags


def align_predictions(true_len, pred_tags):
    if len(pred_tags) < true_len:
        pred_tags += ["O"] * (true_len - len(pred_tags))
    elif len(pred_tags) > true_len:
        pred_tags = pred_tags[:true_len]
    return pred_tags


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate an IOB-format model with vLLM.",
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
        default=1024,
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
        allowed_bio = LENER_ALLOWED_BIO
        DatasetClass = LenerDataset
    else:
        from src.datasets.ulysses import UlyssesDataset
        allowed_bio = ULYSSES_ALLOWED_BIO
        DatasetClass = UlyssesDataset

    # ------------------------------------------------------------------
    # Tokenizer and dataset
    # ------------------------------------------------------------------
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    # Ensure PAD != EOS so generation termination is reliable
    tokenizer.pad_token = "<|endoftext|>"
    tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")

    l = DatasetClass(tokenizer=tokenizer)
    data = l.load_dataset(format="iob")

    split_data = data[args.split]
    print(f"Loaded {args.dataset}/{args.split}: {len(split_data)} examples")

    # The 'prompt' field in IOB format is a list of pre-tokenized input IDs
    raw_prompts = split_data["prompt"]

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
    # Generate (pass pre-tokenized prompt IDs directly to vLLM)
    # ------------------------------------------------------------------
    formatted_prompts = [{"prompt_token_ids": tokens} for tokens in raw_prompts]
    print(f"Generating for {len(formatted_prompts)} prompts...")

    if is_lora:
        outputs = llm.generate(
            prompts=formatted_prompts,
            sampling_params=sampling_params,
            lora_request=LoRARequest("ner_adapter", 1, checkpoint_abs),
        )
    else:
        outputs = llm.generate(prompts=formatted_prompts, sampling_params=sampling_params)

    decoded_texts = [out.outputs[0].text for out in outputs]
    print(f"Generated {len(decoded_texts)} sequences.")

    # ------------------------------------------------------------------
    # Post-process: IOB lines → aligned tag sequences
    # ------------------------------------------------------------------
    ner_feature = data["train"].features["ner_tags"]
    tag_id_to_name = {i: name for i, name in enumerate(ner_feature.feature.names)}

    true_tags_raw = split_data["ner_tags"]

    preds = [parse_llm_output(text) for text in decoded_texts]
    preds = [align_predictions(len(true_tags_raw[i]), pred) for i, pred in enumerate(preds)]

    clean_preds = [
        ["O" if tag not in allowed_bio else tag for tag in seq]
        for seq in preds
    ]
    y_true = [[tag_id_to_name[t] for t in seq] for seq in true_tags_raw]

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------
    report = classification_report(y_true, clean_preds)
    score = f1_score(y_true, clean_preds)
    print("\nClassification Report:\n")
    print(report)
    print(f"F1 Score: {score:.4f}")

    # ------------------------------------------------------------------
    # Save report
    # ------------------------------------------------------------------
    safe_path = args.checkpoint_path.replace("/", "_")
    output_filename = Path(args.output_dir) / f"{safe_path}_{args.split}.txt"
    output_filename.parent.mkdir(parents=True, exist_ok=True)

    with open(output_filename, "w", encoding="utf-8") as f:
        f.write(f"Model: {args.checkpoint_path}\n")
        f.write(f"Dataset: {args.dataset} / {args.split}\n")
        f.write(f"Format: IOB\n")
        f.write("=" * 50 + "\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write(f"\nF1 Score: {score:.4f}\n")

    # Also save raw decoded outputs for inspection
    output_results = Path(args.output_dir) / f"{safe_path}_{args.split}_outputs.txt"
    with open(output_results, "w", encoding="utf-8") as f:
        f.write("\n---\n".join(decoded_texts))

    print(f"\nReport saved to: {output_filename}")
    print(f"Raw outputs saved to: {output_results}")


if __name__ == "__main__":
    main()
