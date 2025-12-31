import sys
import os
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.datasets.lener import LenerDataset
import torch
from tqdm.autonotebook import tqdm
from seqeval.metrics import classification_report, f1_score
import argparse
from pathlib import Path 
import pickle

os.environ['HF_HOME'] = '/work1/lgarcia/pedrobpio/HF_files'

parser = argparse.ArgumentParser(description="Dynamically import checkpoint path and model name.")
parser.add_argument('model_info',
                    metavar='model path and model name',
                    type=str,
                    nargs='+',
                    help='Two arguments; model_path model_name')

args = parser.parse_args()

MODEL_PATH = f"notebooks/outputs/checkpoints/{args.model_info[0]}"
MODEL_NAME = args.model_info[1]
print(f"Model path: {MODEL_PATH}")
print(f"Model name: {MODEL_NAME}")

attn_impl = "flash_attention_2"
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH,
                                            dtype="auto", # Ou torch.bfloat16/torch.float16 explicitamente
                                            # device_map="auto",
                                            attn_implementation="flash_attention_2",          # Often needed for custom architectures like Qwen
                                            )
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
device = f'cuda:{args.model_info[2]}' if torch.cuda.is_available() else "cpu"
model.to(device)

head_dim = model.config.hidden_size // model.config.num_attention_heads
print(f"🔍 Head Dimension do Modelo: {head_dim}")

model = torch.compile(model)

print(model.device)

### load data
l = LenerDataset(tokenizer=tokenizer)
pickle_data = 'lener_iob.pkl'
try:
    with open(pickle_data, 'rb') as file:
        l = pickle.load(file)
        data = l.dataset
        print('loaded data successfully')
except FileNotFoundError:

    data = l.load_dataset()
    with open(pickle_data, 'wb') as file:
        pickle.dump(l, file)






# 1. Configuration
batch_size = int(args.model_info[3])  # Adjust based on your GPU VRAM
device = model.device

# Ensure we have a valid pad token
pad_token_id = tokenizer.pad_token_id
if pad_token_id is None:
    pad_token_id = tokenizer.eos_token_id

# 2. Get the list of tokenized prompts
# Assuming raw_prompts is a list of lists: [[101, 20, ...], [101, 55, ...]]
raw_prompts = data['validation']['prompt']
# raw_prompts = data['test']['prompt'] 

all_generated_ids = []

# 3. Iterate in batches
for i in tqdm(range(0, len(raw_prompts), batch_size), desc="Generating"):
    # Slice the current batch
    batch_sequences = raw_prompts[i : i + batch_size]
    
    # --- Dynamic Padding Logic (Applied per batch) ---
    # Find the max length needed strictly for THIS batch
    batch_max_len = max(len(seq) for seq in batch_sequences)
    
    padded_input_ids = []
    attention_masks = []
    
    for seq in batch_sequences:
        num_pads = batch_max_len - len(seq)
        
        # LEFT Padding (Crucial for generation)
        padded_row = [pad_token_id] * num_pads + seq
        
        # Attention Mask (0 for pad, 1 for real)
        mask_row = [0] * num_pads + [1] * len(seq)
        
        padded_input_ids.append(padded_row)
        attention_masks.append(mask_row)
        
    # Convert to tensors
    input_ids = torch.tensor(padded_input_ids, dtype=torch.long).to(device)
    attention_mask = torch.tensor(attention_masks, dtype=torch.long).to(device)
    
    # 4. Generate
    generation_config = {
        "do_sample": False,
        # "temperature": 0.01,
        # "top_p": 0.9,
        "repetition_penalty": 1.0,
        "max_new_tokens": 1024,
        "pad_token_id": pad_token_id,
        "eos_token_id": tokenizer.eos_token_id
    }

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
    
    # 5. Store results
    # We convert tensors back to lists to save CPU RAM, or keep as CPU tensors
    all_generated_ids.extend(outputs.cpu().tolist())

    # Optional: Clear cache if running on low-memory GPU
    torch.cuda.empty_cache()
print(f"Generated {len(all_generated_ids)} sequences.")

# 6. Decode (Convert IDs back to text)b when needed
# You can do this here or process the IDs directly
decoded_texts = tokenizer.batch_decode(all_generated_ids, skip_special_tokens=True)




# from datasets import load_dataset

# 1. Load Lener-Br to get the ID-to-Label mapping
# dataset = load_dataset("lener_br", trust_remote_code=True)

# Extract the feature mapping (0 -> 'O', 1 -> 'B-ORGANIZACAO', etc.)
# Note: Lener-BR usually uses unaccented caps (ORGANIZACAO, not ORGANIZAÇÃO)
# label_list = data['train'].features['ner_tags'].feature.names
# id2label = {i: label for i, label in enumerate(label_list)}

def parse_llm_output(llm_text):
    """
    Parses lines like "Súmula:B-JURISPRUDENCIA" into a list of tags.
    """
    # def _process_output(pred_tags, lener_instance):
    #     return [lener_instance.name_to_tag_id.get(tag) for tag in pred_tags]
    
    pred_tags = []
    llm_text = llm_text.split("Resposta:\n")[-1]
    lines = llm_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if not line: continue
            
        # Split by the LAST colon to separate Token from Tag
        # We use rsplit because the token itself might contain a colon (e.g., 10:30)
        if ':' in line:
            parts = line.rsplit(':', 1)
            tag = parts[-1].strip()
            pred_tags.append(tag)
        else:
            # Fallback if format is broken (LLM hallucination)
            pred_tags.append('O')
    
    return pred_tags

def align_predictions(true_len, pred_tags):
    """
    Ensures predicted tags match the length of the ground truth.
    """
    # If LLM produced too few tags, pad with 'O'
    if len(pred_tags) < true_len:
        pred_tags += ['O'] * (true_len - len(pred_tags))
        
    # If LLM produced too many tags, truncate
    elif len(pred_tags) > true_len:
        pred_tags = pred_tags[:true_len]
        
    return pred_tags

preds=[parse_llm_output(output) for output in decoded_texts]
true_tags = data['validation']['ner_tags']
# true_tags = data['test']['ner_tags']
# pred = align_predictions(len(true_tags), pred)
preds = [align_predictions(len(true_tags[idx]), pred) for idx, pred in enumerate(preds)]

allowed_tags = ["O",
              "B-PESSOA", "I-PESSOA",
              "B-ORGANIZACAO","I-ORGANIZACAO",
              "B-LOCAL", "I-LOCAL",
              "B-TEMPO", "I-TEMPO",
              "B-LEGISLACAO", "I-LEGISLACAO",
              "B-JURISPRUDENCIA", "I-JURISPRUDENCIA"]
clean_preds = [
    ['O' if tag not in allowed_tags else tag for tag in seq] 
    for seq in preds
]
y_true = [[l.tag_id_to_name[tag] for tag in true_tag] for true_tag in true_tags ]
print(classification_report(y_true, clean_preds))

output_filename = f"./outputs/reports/{args.model_info[0]}.txt"

output_path = Path(output_filename)
output_dir = output_path.parent

output_dir.mkdir(parents=True, exist_ok=True)
print(f"Ensured directory exists: {output_dir}")

try:
    with open(output_filename, 'w', encoding='utf-8') as f:
        # 3. Write the stored metrics to the file
        f.write("📊 Classification Report (BIO Format):\n")
        f.write("=" * 40 + "\n") # Optional separator
        f.write(classification_report(y_true, clean_preds))
        f.write("\n\n") # Add some space before the next metric

        f.write("=" * 40 + "\n") # Optional separator
        # Use an f-string to format the F1 score nicely
        f.write(f"🔁 F1 Score: {f1_score(y_true, clean_preds)}\n") # Format F1 to 4 decimal places
        f.write("=" * 40 + "\n") # Optional separator

    print(f"\n✅ Metrics successfully saved to: {output_filename}")

except Exception as e:
    print(f"\n❌ Error saving metrics to file: {e}")
