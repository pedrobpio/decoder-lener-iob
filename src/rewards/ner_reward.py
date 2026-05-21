import re
import logging

logger = logging.getLogger(__name__)

LENER_TAGS = ["ORGANIZACAO", "PESSOA", "TEMPO", "LOCAL", "LEGISLACAO", "JURISPRUDENCIA"]
ULYSSES_TAGS = ["ORGANIZACAO", "PESSOA", "DATA", "LOCAL", "EVENTO", "FUNDAMENTO", "PRODUTODELEI"]

def parse_ner_output(text, valid_tags=None):
    """
    Parses model output in entity-extraction format: "TAG: Entity String; ...".
    Strips the "Resposta:" header if present.
    Entity spans are lowercased for comparison to avoid surface-form casing mismatches.

    valid_tags: list of tag strings to match. Defaults to LENER_TAGS.
    Returns a set of (tag, entity_span_lowercase) tuples.
    """
    if "Resposta:" in text:
        text = text.split("Resposta:")[-1]

    tags = valid_tags or LENER_TAGS
    pattern = r"(" + "|".join(tags) + r"):\s*([^;\n]+)"
    matches = re.findall(pattern, text)

    return set((tag.strip(), value.strip().lower()) for tag, value in matches)


def ner_reward_func(completions, **kwargs):
    """
    GRPO reward function for NER in entity-extraction format.

    completions: list of generated strings
    kwargs: must contain 'ground_truth' (list of reference entity strings)
            may contain 'valid_tags' (list of tag strings, defaults to LENER_TAGS)

    Reward range: [-1.0, 1.0]
      -1.0  model hallucinated entities when ground truth is empty
      -0.2  model produced no entities when ground truth is non-empty
       0.0  model produced wrong entities (F1 = 0)
       0..1 F1 score for partial matches
       1.0  perfect exact match (all entities correct, none extra)
    """
    rewards = []
    ground_truth_list = kwargs.get("ground_truth", [])
    valid_tags = kwargs.get("valid_tags", None)

    for completion, ground_truth_str in zip(completions, ground_truth_list):
        pred_entities = parse_ner_output(completion, valid_tags=valid_tags)
        true_entities = parse_ner_output(ground_truth_str, valid_tags=valid_tags)

        # Both empty: model correctly predicted no entities
        if not true_entities and not pred_entities:
            rewards.append(1.0)
            continue

        # Hallucination: model predicted entities when there are none
        if not true_entities:
            rewards.append(-1.0)
            continue

        # Model predicted nothing when entities exist
        if not pred_entities:
            rewards.append(-0.2)
            continue

        # Perfect match: all entities correct, no extras
        if pred_entities == true_entities:
            rewards.append(1.0)
            continue

        # Partial match: compute F1
        correct_matches = pred_entities.intersection(true_entities)
        precision = len(correct_matches) / len(pred_entities)
        recall = len(correct_matches) / len(true_entities)

        if (precision + recall) == 0:
            rewards.append(0.0)
        else:
            f1 = 2 * (precision * recall) / (precision + recall)
            rewards.append(f1)

    return rewards
