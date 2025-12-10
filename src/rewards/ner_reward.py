import re
import logging

logger = logging.getLogger(__name__)

def parse_ner_output(text):
    """
    Parses the model output. 
    Assumes the format from your 'format_example' method: "TAG: Entity String" separated by ; or newlines.
    """
    # Remove the "Resposta:" header if present to avoid confusion
    if "Resposta:" in text:
        text = text.split("Resposta:")[-1]
    
    # Pattern to capture "TAG: Value"
    # Adjust regex based on exactly how your model outputs data
    pattern = r"(ORGANIZACAO|PESSOA|TEMPO|LOCAL|LEGISLACAO|JURISPRUDENCIA):\s*([^;\n]+)"
    matches = re.findall(pattern, text)
    
    # Return a set of tuples for easy comparison: {('PESSOA', 'João da Silva'), ...}
    return set((tag.strip(), value.strip()) for tag, value in matches)

def ner_reward_func(completions, **kwargs):
    """
    GRPO calls this function.
    completions: list of strings (generated text)
    kwargs: contains 'ground_truth' (the reference entities)
    """
    rewards = []
    ground_truth_list = kwargs.get("ground_truth", [])

    for completion, ground_truth_str in zip(completions, ground_truth_list):
        # 1. Parse the generated output
        pred_entities = parse_ner_output(completion)
        
        # 2. Parse the ground truth (assuming it's passed as a string in the dataset)
        true_entities = parse_ner_output(ground_truth_str)

        # 3. Calculate Score (F1-style or Recall-heavy)
        if not true_entities and not pred_entities:
            # Both empty = Correct
            rewards.append(1.0)
            continue
            
        if not true_entities:
            # Model halluncinated entities where there are none
            rewards.append(-0.5) 
            continue

        # Intersection
        correct_matches = pred_entities.intersection(true_entities)
        
        # Metrics
        precision = len(correct_matches) / len(pred_entities) if pred_entities else 0.0
        recall = len(correct_matches) / len(true_entities) if true_entities else 0.0
        
        # F1 Score
        if (precision + recall) == 0:
            f1 = 0.0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
        
        # Bonus: If strict exact match
        if pred_entities == true_entities:
            f1 += 0.5 # Boost for perfection

        rewards.append(f1)

    return rewards