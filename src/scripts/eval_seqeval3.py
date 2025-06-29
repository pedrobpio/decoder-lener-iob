from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
from tqdm import tqdm
import torch
from seqeval.metrics import classification_report, f1_score
import re
import argparse
import string 
from pathlib import Path 
import sys

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt_tab')

parser = argparse.ArgumentParser(description="Dynamically import checkpoint path and model name.")
parser.add_argument('model_info',
                    metavar='model path and model name',
                    type=str,
                    nargs='+',
                    help='Two arguments; model_path model_name')

args = parser.parse_args()

MODEL_PATH = f"./outputs/checkpoints/{args.model_info[0]}"
MODEL_NAME = args.model_info[1]
print(f"Model path: {MODEL_PATH}")
print(f"Model name: {MODEL_NAME}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, attn_implementation="eager")
model.eval()
device = f'cuda:{args.model_info[2]}' if torch.cuda.is_available() else "cpu"
model.to(device)

dataset = load_dataset("peluz/lener_br", split="validation[:]")
# dataset = load_dataset("peluz/lener_br", split="test[:]")

ner_labels = dataset.features["ner_tags"].feature.names
     

context_index = args.model_info[3]
context = {'les':  """Você é um especialista jurídico responsável por identificar entidades de LEGISLACAO em textos.        
As entidades de LEGISLACAO se referem a Atos de Lei, como leis, decretos, portarias, etc.
segue o texto\n""",
        'juri': """Você é um especialista jurídico responsável por identificar entidades de JURISPRUDENCIA em textos.        
As entidades de JURISPRUDENCIA se referem a decisões relativas a casos legais anteriores.
segue o texto\n""",
        'org':"""Você é um especialista jurídico responsável por identificar entidades de ORGANIZACAO em textos.        
As entidades de ORGANIZACAO são entidades que representam organizações, como empresas, instituições governamentais, ONGs, etc.
segue o texto\n""",
        'pes':"""Você é um especialista jurídico responsável por identificar entidades de PESSOA em textos.        
As entidades de PESSOA são nomes de pessoas físicas.
segue o texto\n""",
        'loc':"""Você é um especialista jurídico responsável por identificar entidades de LOCAL em textos.        
As entidades de LOCAL são entidades que representam lugares geográficos, como cidades, países, estados, endereços, etc.
segue o texto\n""",
        'temp': """Você é um especialista jurídico responsável por identificar entidades de TEMPO em textos.        
As entidades de TEMPO são entidades que expressam informações temporais, como datas, horários, períodos, etc.
segue o texto\n""",
        'full':"""Você é um especialista jurídico responsável por identificar entidades em textos.        
As entidades que você deve identificar são:

- ORGANIZAÇÃO: Refere-se a entidades que representam organizações, como empresas, instituições governamentais, ONGs, etc.
- PESSOA: Designa entidades que são nomes de pessoas físicas.
- TEMPO: Marca entidades que expressam informações temporais, como datas, horários, períodos, etc.
- LOCAL: Indica entidades que representam lugares geográficos, como cidades, países, estados, endereços, etc.
- LEGISLAÇÃO: Identifica entidades que correspondem a Atos de Lei, como leis, decretos, portarias, etc.
- JURISPRUDÊNCIA: Assinala entidades que se referem a decisões relativas a casos legais.      

segue o texto\n"""
}
chosen_context = context[context_index]

print(f'context -> {chosen_context}')

def build_prompt(example, chosen_context):
    context_prompt = (
        chosen_context
        )
    sentence = " ".join(example["tokens"])
    return f"{context_prompt}Texto: {sentence}\nEntidades:"

def extract_entities(text, valid_labels):
    """
    Extracts entities from text based on a list of valid labels.

    Args:
        text (str): The input text containing an "Entidades:" section.
        valid_labels (list): A list of strings representing the valid entity labels
                               (e.g., ['LEGISLACAO', 'PESSOA', 'LOCAL']).

    Returns:
        list: A list of tuples, where each tuple is (LABEL, value).
              Returns an empty list if "Entidades:" is not found or no entities match.
    """
    # 1. Find the text block after "Entidades:" (case-insensitive)
    # re.DOTALL allows '.' to match newline characters if entities span lines
    match = re.search(r"Entidades:\s*(.*)", text, re.IGNORECASE | re.DOTALL)
    if not match:
        return []

    entity_string = match.group(1).strip()
    if not entity_string: # Handle case where "Entidades:" is present but empty
        return []

    # 2. Prepare the regex pattern based on valid labels
    # Escape labels in case they contain special regex characters (like '.')
    escaped_labels = [re.escape(label) for label in valid_labels]
    # Create a pattern that matches any of the valid labels (case-insensitive)
    label_pattern = "|".join(escaped_labels)

    # 3. Build the main regex to find all label: value pairs
    # Pattern breakdown:
    # ({label_pattern})   : Capture group 1: One of the valid labels (case-insensitive)
    # :                   : Match the literal colon
    # \s* : Match optional whitespace after the colon
    # (.*?)               : Capture group 2: The value (non-greedy match)
    # (?=                 : Positive lookahead assertion (checks ahead without consuming chars)
    #   \s* : Optional whitespace before the next potential label
    #   (?:{label_pattern}): : A non-capturing group matching the start of the *next* label and colon
    #   |                 : OR
    #   $                 : The end of the string
    # )
    # This ensures the value (.*?) stops *before* the next known label or the end of the string.
    # re.IGNORECASE applies to the label matching part.
    # re.DOTALL allows '.*?' to match across newlines if needed.
    pattern = rf"({label_pattern}):\s*(.*?)(?=\s*(?:{label_pattern}):|$)"

    # 4. Find all matches using findall
    # findall returns a list of tuples, where each tuple contains the captured groups.
    # In this case: [ (label1, value1), (label2, value2), ... ]
    found_entities = re.findall(pattern, entity_string, re.IGNORECASE | re.DOTALL)

    # 5. Format the results: Uppercase label, stripped value
    entities = []
    for label, value in found_entities:
        entities.append((label.strip().upper(), value.strip()))

    return entities


def tokenize_text(text):
    """Splits text into words (sequences of letters/numbers/_) and punctuation/symbols.
    This method appears consistent with the observed tokenization in LeNER-Br examples."""
    # \w+ matches word characters (alphanumeric + underscore)
    # [^\w\s] matches any character that is not a word character and not whitespace (i.e., punctuation)
    # regex = r'\w+(?:[-\.\/º]\w*)*\.?|[^\w\s]+'
    # tokens = re.findall(regex, text)
    return word_tokenize(text)

def preprocess_entity_text(text):
    """Removes content after the first newline and any trailing punctuation."""
    # 1. Remove content after '\n'
    processed_text = text.split('\n', 1)[0]

    # 2. Remove trailing punctuation repeatedly
    #    (e.g., handles cases like "text..", "text?!")
    while processed_text and processed_text[-1] in string.punctuation:
        processed_text = processed_text[:-1]
        
    # 3. Optional: Remove leading/trailing whitespace that might remain
    processed_text = processed_text.strip() 

    return processed_text

def filter_and_merge_entities(entities):
    """
    Preprocesses entity text, then filters to remove duplicates and 
    contained entities.
    
    Preprocessing steps:
    1. Removes text after the first newline ('\n').
    2. Removes trailing punctuation.
    """
    if not entities:
        return []

    # --- Step 1: Preprocess all entity texts ---
    preprocessed_entities = []
    for label, text in entities:
        cleaned_text = preprocess_entity_text(text)
        # Keep the entity even if the cleaned_text becomes empty
        preprocessed_entities.append((label, cleaned_text))
    
    # --- Step 2: Remove exact duplicates after preprocessing ---
    # Using set() automatically handles duplicates based on the (label, cleaned_text) tuple
    unique_entities = list(set(preprocessed_entities))

    # --- Step 3: Handle simple cases ---
    if len(unique_entities) <= 1:
        unique_entities.sort() # Sort for consistent output
        return unique_entities

    # --- Step 4: Filter subsumed entities (using preprocessed text) ---
    final_entities = []
    # It's often good practice to sort before the nested loop, 
    # although the original code sorted at the end. Sorting here doesn't change correctness.
    # unique_entities.sort() 

    for i in range(len(unique_entities)):
        entity_a = unique_entities[i]
        label_a, text_a = entity_a
        is_subsumed_by_another = False

        for j in range(len(unique_entities)):
            if i == j: continue # Don't compare an entity with itself
            
            entity_b = unique_entities[j]
            label_b, text_b = entity_b

            # Check for subsumption:
            # - Same label
            # - Text A is not identical to Text B
            # - Text A is contained within Text B 
            #   (e.g., "Bank" is in "Bank of America")
            if label_a == label_b and text_a != text_b and text_a in text_b:
                 # We should only consider non-empty text_a as being subsumed by a longer string.
                 # An empty string '' is technically 'in' any other string, 
                 # but we usually don't want to remove ('LABEL', '') just because ('LABEL', 'something') exists.
                 # However, the original logic *would* remove the empty string version. 
                 # Let's stick to the direct interpretation of the original logic for now.
                 # If text_a is '', it will be subsumed by any non-empty text_b with the same label.
                is_subsumed_by_another = True
                break # Found a entity (B) that subsumes A, no need to check further for A

        if not is_subsumed_by_another:
            final_entities.append(entity_a)

    # --- Step 5: Sort the final list ---
    final_entities.sort() # Sort based on label, then text
    return final_entities

def find_prefix_before_match(text: str) -> str:

    # --- Input Validation ---
    # Return original text if no labels or text is provided, preventing errors later.
    if not text:
        return text
    known_labels = ['LEGISLACAO', 'JURISPRUDENCIA', 'TEMPO', 'PESSOA', 'ORGANIZACAO', 'LOCAL']
    # --- 1. Generate all possible match patterns ---
    # We use a set to automatically handle duplicate patterns efficiently.
    match_patterns = set()
    for label in known_labels:
        # Only consider labels that are at least 3 characters long,
        # as prefixes must be at least 3 characters.
        if len(label) >= 3:
            # Generate prefixes from length 3 up to the full label length.
            # range(3, len(label) + 1) ensures we get slices like label[:3], label[:4], ..., label[:len(label)]
            for i in range(3, len(label) + 1):
                match_patterns.add(label[:i])
        # Note: Full labels shorter than 3 chars are ignored based on the rule
        # that subwords (prefixes) must be >= 3 chars. If you wanted to match
        # exact full words shorter than 3, you'd add an `else: match_patterns.add(label)` here.

    # --- 2. Find the earliest occurrence of any pattern in the text ---
    # Initialize with a value guaranteed to be larger than any valid index.
    first_match_index = sys.maxsize

    # Iterate through all generated patterns (full words >=3 chars and prefixes >= 3 chars)
    for pattern in match_patterns:
        try:
            # string.find() performs a case-sensitive search.
            # It returns the starting index of the first occurrence or -1 if not found.
            index = text.find(pattern)

            # Check if the pattern was found (index is not -1)
            # and if this occurrence is earlier than the earliest one found so far.
            if index != -1 and index < first_match_index:
                first_match_index = index # Update the earliest index found
        except TypeError:
             # This handles potential (though unlikely for string.find) type errors gracefully.
             print(f"Warning: Type error encountered while searching for pattern '{pattern}'")
             pass # Continue searching with other patterns

    # --- 3. Return the result based on whether a match was found ---
    # If first_match_index was updated from its initial large value, a match was found.
    if first_match_index != sys.maxsize:
        # Return the slice of the string from the beginning up to the start of the match.
        return text[:first_match_index]
    else:
        # No match was found among any of the patterns, return the original string.
        return text

# --- NEW: Helper function to clean entity text ---
def clean_entity_text(text):
    """Removes common trailing punctuation and whitespace from entity text."""
    # Define characters to strip from the end
    chars_to_strip = string.punctuation + string.whitespace
    return text.rstrip(chars_to_strip)

# --- MODIFIED: BIO Tagging Function (incorporates cleaning) ---
def match_entities_to_tokens(tokens, filtered_entities):
    """
    Matches extracted entities to tokens and generates BIO tags.
    Assumes consistent tokenization and pre-filtered entities.
    Includes cleaning of entity text before matching.
    """
    if not tokens:
        return []

    bio_tags = ["O"] * len(tokens)
    num_tokens = len(tokens)

    # Pre-process entities: clean text and tokenize
    tokenized_entities = []
    for label, ent_text in filtered_entities:
        
        filtred_ent_text = find_prefix_before_match(ent_text)
        # *** Clean the entity text before tokenizing ***
        cleaned_ent_text = clean_entity_text(filtred_ent_text)
        # Tokenize the *cleaned* text
        ent_tokens = tokenize_text(cleaned_ent_text)

        if ent_tokens:
             tokenized_entities.append({
                 'label': label,
                 'original_text': ent_text, # Keep original for reference if needed
                 'cleaned_text': cleaned_ent_text,
                 'tokens': ent_tokens # Tokens from the cleaned text
             })

    # Sort by token length descending (prioritizes longer matches)
    tokenized_entities.sort(key=lambda x: len(x['tokens']), reverse=True)

    # --- Optional Debugging ---
    # print("\nCleaned and Tokenized Entities for Matching:")
    # for te in tokenized_entities:
    #     print(f"- {te['label']}: {te['tokens']} (from cleaned: '{te['cleaned_text']}')")
    # --- End Debugging ---

    for entity_data in tokenized_entities:
        label = entity_data['label']
        ent_tokens = entity_data['tokens'] # Use the tokens from the cleaned text
        num_ent_tokens = len(ent_tokens)

        if num_ent_tokens == 0: continue # Skip if cleaning resulted in empty tokens

        # Iterate through possible start positions in the main token list
        for i in range(num_tokens - num_ent_tokens + 1):
            token_slice = tokens[i : i + num_ent_tokens]
            # Compare slice with the entity tokens (derived from *cleaned* text)
            if token_slice == ent_tokens:
                # Match found! Check if span is clear before tagging.
                is_span_clear = all(bio_tags[i + j] == "O" for j in range(num_ent_tokens))
                if is_span_clear:
                    bio_tags[i] = f"B-{label.upper()}"
                    for j in range(1, num_ent_tokens):
                        bio_tags[i + j] = f"I-{label.upper()}"
                    # If a match is tagged, move to the next entity
                    # (We assume non-overlapping based on filtering/sorting,
                    # but this break makes it slightly more efficient if an entity
                    # text could appear multiple times identically)
                    # However, let's keep searching in case the SAME cleaned text
                    # appears multiple times in the source doc. So, NO break here.
    return bio_tags



y_true = []
y_pred = []
known_labels = ['LEGISLACAO', 'JURISPRUDENCIA', 'TEMPO', 'PESSOA', 'ORGANIZACAO', 'LOCAL']
prompt_batch_size = 32
output = []
total = len(dataset)
for i in tqdm(range(0, len(dataset), prompt_batch_size)):
    prompts = []
    batch_examples = dataset[i : i + prompt_batch_size]
    
    for j in tqdm(range(prompt_batch_size)):
        if total > 0:
            example = {"tokens": batch_examples['tokens'][j], "ner_tags": batch_examples['ner_tags'][j]}
        else:
            continue
        tokens = example["tokens"]
        gold_tags = [ner_labels[tag] for tag in example["ner_tags"]]

        prompt = build_prompt(example, chosen_context)
        prompts.append(prompt)
        total -=1
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(prompts, return_tensors="pt", padding=True,truncation=True).to(device)
    
    with torch.no_grad():
            output_ids = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                max_new_tokens=128,
                do_sample=False,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                
            )  
    output += output_ids  

for i, output in enumerate(output):
    tokens = dataset[i]['tokens']
    gold_tags = [ner_labels[tag] for tag in dataset[i]["ner_tags"]]
    output_text = tokenizer.decode(output, skip_special_tokens=True)
    generated_entities = extract_entities(output_text, known_labels)
    generated_entities = filter_and_merge_entities(generated_entities)
    pred_tags = match_entities_to_tokens(tokens, generated_entities)

    y_true.append(gold_tags)
    y_pred.append(pred_tags)

# Print metrics
print("\n📊 Classification Report (BIO Format):\n")
print(classification_report(y_true, y_pred))
print("🔁 F1 Score:", f1_score(y_true, y_pred))

# output_filename = f"./outputs/reports/{args.model_info[0]}_test.txt"
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
        f.write(classification_report(y_true, y_pred))
        f.write("\n\n") # Add some space before the next metric

        f.write("=" * 40 + "\n") # Optional separator
        # Use an f-string to format the F1 score nicely
        f.write(f"🔁 F1 Score: {f1_score(y_true, y_pred)}\n") # Format F1 to 4 decimal places
        f.write("=" * 40 + "\n") # Optional separator

    print(f"\n✅ Metrics successfully saved to: {output_filename}")

except Exception as e:
    print(f"\n❌ Error saving metrics to file: {e}")
