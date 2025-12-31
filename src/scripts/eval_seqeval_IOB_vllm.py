import sys
import os
import argparse
from pathlib import Path 
import pickle
import json
import torch
from transformers import AutoTokenizer
from src.datasets.lener import LenerDataset
from src.datasets.ulysses import UlyssesDataset
from seqeval.metrics import classification_report, f1_score
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest 

# Set HF Home globally so it applies to all processes
os.environ['HF_HOME'] = '/work1/lgarcia/pedrobpio/HF_files'

def parse_llm_output(llm_text):
    pred_tags = []
    if "Resposta:\n" in llm_text:
        llm_text = llm_text.split("Resposta:\n")[-1]
    
    lines = llm_text.strip().split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        if ':' in line:
            parts = line.rsplit(':', 1)
            tag = parts[-1].strip()
            pred_tags.append(tag)
        else:
            pred_tags.append('O')
    return pred_tags

def align_predictions(true_len, pred_tags):
    if len(pred_tags) < true_len:
        pred_tags += ['O'] * (true_len - len(pred_tags))
    elif len(pred_tags) > true_len:
        pred_tags = pred_tags[:true_len]
    return pred_tags

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('model_info', nargs='+', help='model_path model_name gpu_id batch_size')
    args = parser.parse_args()

    # --- 1. Fix Path to Absolute ---
    relative_path = f"notebooks/outputs/checkpoints/{args.model_info[0]}"
    MODEL_PATH = str(Path(relative_path).resolve())

    MODEL_NAME = args.model_info[1]
    GPU_ID = args.model_info[2]

    print(f"📍 Absolute Model Path: {MODEL_PATH}")
    print(f"Target GPU: {GPU_ID}")

    # Set GPU Visibility
    os.environ["CUDA_VISIBLE_DEVICES"] = GPU_ID

    # --- 2. Check if this is a LoRA Adapter ---
    is_lora = False
    base_model_path = MODEL_PATH 

    if os.path.exists(os.path.join(MODEL_PATH, "adapter_config.json")):
        print("⚠️ Detected LoRA Adapter checkpoint (not a full model).")
        is_lora = True
        
        # Read the base model name from the adapter config
        with open(os.path.join(MODEL_PATH, "adapter_config.json"), 'r') as f:
            adapter_conf = json.load(f)
            base_model_path = adapter_conf.get("base_model_name_or_path")
        
        print(f"   -> Loading Base Model: {base_model_path}")
        print(f"   -> Will apply LoRA from: {MODEL_PATH}")

    # --- Load Tokenizer & Dataset ---
    tokenizer_path = base_model_path if is_lora else MODEL_PATH
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # l = LenerDataset(tokenizer=tokenizer)
    # pickle_data = 'lener_iob.pkl'

    l = UlyssesDataset(tokenizer=tokenizer)
    pickle_data = 'ulysses_iob.pkl'

    print("Loading Data...")
    try:
        with open(pickle_data, 'rb') as file:
            l = pickle.load(file)
            data = l.dataset
            print('loaded data successfully')
    except FileNotFoundError:
        data = l.load_dataset()
        with open(pickle_data, 'wb') as file:
            pickle.dump(l, file)

    # --- vLLM Initialization ---
    print("Initializing vLLM...")

    llm = LLM(
        model=base_model_path, 
        tokenizer=tokenizer_path,
        dtype="auto",
        gpu_memory_utilization=0.90, 
        tensor_parallel_size=1,
        trust_remote_code=True,
        enable_lora=is_lora, 
        max_lora_rank=64 if is_lora else 16 
    )

    sampling_params = SamplingParams(
        seed=42,
        temperature=0.0,
        max_tokens=1024,
        repetition_penalty=1.0,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else []
    )

    # --- Generation ---
    # raw_prompts = data['validation']['prompt']
    raw_prompts = data['test']['prompt']
    print(f"Starting generation for {len(raw_prompts)} prompts...")
    formatted_prompts = [
    {"prompt_token_ids": tokens} for tokens in raw_prompts
]
    if is_lora:
        outputs = llm.generate(
            prompts=formatted_prompts, 
            sampling_params=sampling_params,
            lora_request=LoRARequest("lener_adapter", 1, MODEL_PATH)
        )
    else:
        outputs = llm.generate(
            prompts=formatted_prompts, 
            sampling_params=sampling_params
        )

    decoded_texts = [output.outputs[0].text for output in outputs]
    print(f"Generated {len(decoded_texts)} sequences.")

    # --- Post-Processing & Metrics ---
    print("Calculating Metrics...")

    preds = [parse_llm_output(output) for output in decoded_texts]
    # true_tags = data['validation']['ner_tags']
    true_tags = data['test']['ner_tags']

    preds = [align_predictions(len(true_tags[idx]), pred) for idx, pred in enumerate(preds)]

    # allowed_tags = ["O", "B-PESSOA", "I-PESSOA", "B-ORGANIZACAO","I-ORGANIZACAO", "B-LOCAL", "I-LOCAL", "B-TEMPO", "I-TEMPO", "B-LEGISLACAO", "I-LEGISLACAO", "B-JURISPRUDENCIA", "I-JURISPRUDENCIA"]
    allowed_tags = ['O', 'B-DATA', 'I-DATA', 'B-EVENTO', 'I-EVENTO', 'B-FUNDAMENTO', 'I-FUNDAMENTO', 'B-LOCAL', 'I-LOCAL', 'B-ORGANIZACAO', 'I-ORGANIZACAO', 'B-PESSOA', 'I-PESSOA', 'B-PRODUTODELEI', 'I-PRODUTODELEI']
    clean_preds = [['O' if tag not in allowed_tags else tag for tag in seq] for seq in preds]
    y_true = [[l.tag_id_to_name[tag] for tag in true_tag] for true_tag in true_tags]

    report = classification_report(y_true, clean_preds)
    print(report)

    # --- Save to File ---
    # output_filename = f"./outputs/reports/{args.model_info[0].split('/')[-1]}_(1).txt" 
    output_filename = f"./outputs/reports/{args.model_info[0]}_test.txt"
    output_filename_results = f"./outputs/reports/{args.model_info[0]}_test_results.txt"
    output_path = Path(output_filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_filename_results, 'w', encoding='utf-8') as f:
        f.write('\n'.join(decoded_texts))

    with open(output_filename, 'w', encoding='utf-8') as f:
        f.write("📊 Classification Report (BIO Format):\n")
        f.write(report)
        f.write(f"\n🔁 F1 Score: {f1_score(y_true, clean_preds)}\n") 
    print(f"\n✅ Metrics successfully saved to: {output_filename}")

if __name__ == "__main__":
    main()