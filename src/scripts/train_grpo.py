"""
CLI training script for GRPO fine-tuning.

Follows the same dynamic import pattern as train.py so the same dataset,
model, and LoRA adapter classes can be reused without modification.

Usage example (LeNER-BR, Qwen3-8B, high_rank_64 LoRA, from an SFT checkpoint):

    python src/scripts/train_grpo.py \\
        lener.LenerDataset \\
        qwen3.Qwen3 \\
        lora.LoraAdapter \\
        Qwen3-8B_high_rank_64_grpo \\
        --sft_checkpoint ./outputs/checkpoints/lener_br/Qwen3-8B_high_rank_64/checkpoint-2450 \\
        --lora_preset high_rank_64 \\
        --model_name Qwen/Qwen3-8B \\
        --num_generations 8 \\
        --epochs 3

The SFT checkpoint is highly recommended. GRPO on a base model produces very
noisy reward signals early in training because the model does not yet know the
expected output format.
"""

import sys
import importlib
import argparse
import torch

sys.path.append('..')


def import_class_from_string(full_class_string):
    try:
        module_path, class_name = full_class_string.rsplit('.', 1)
    except ValueError:
        print(f"Error: Invalid format '{full_class_string}'. Expected 'module_name.ClassName'.")
        return None
    try:
        module = importlib.import_module(module_path)
        class_obj = getattr(module, class_name)
        print(f"- Found class: {class_name} in module: {module_path}")
        return class_obj
    except ImportError:
        print(f"Error: Could not import module '{module_path}'.")
        return None
    except AttributeError:
        print(f"Error: Class '{class_name}' not found in module '{module_path}'.")
        return None
    except Exception as e:
        print(f"Unexpected error importing {full_class_string}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="GRPO fine-tuning with dynamic class imports.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'full_class_names',
        metavar='module.ClassName',
        type=str,
        nargs='+',
        help=(
            "Four positional args in order: dataset, model, peft_config, checkpoint_name. "
            "E.g.: lener.LenerDataset qwen3.Qwen3 lora.LoraAdapter my_run"
        ),
    )
    parser.add_argument(
        '--model_name',
        type=str,
        default="Qwen/Qwen3-8B",
        help="HuggingFace model identifier or local path.",
    )
    parser.add_argument(
        '--sft_checkpoint',
        type=str,
        default=None,
        help=(
            "Path to a LoRA adapter checkpoint from SFT training. "
            "The model is warm-started from this adapter before GRPO begins. "
            "Strongly recommended — omit only to train from the base model."
        ),
    )
    parser.add_argument(
        '--lora_preset',
        type=str,
        default="high_rank_64",
        help="LoRA preset name as defined in src/peft_configs/lora.py.",
    )
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--grad_accum', type=int, default=8)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--num_generations', type=int, default=8)
    parser.add_argument('--max_prompt_length', type=int, default=1024)
    parser.add_argument('--max_completion_length', type=int, default=256)
    parser.add_argument('--beta', type=float, default=0.04)
    parser.add_argument('--save_steps', type=int, default=50)
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help="Device string (e.g. 'cuda:0'). Auto-detected if omitted.",
    )

    args = parser.parse_args()

    if len(args.full_class_names) != 4:
        parser.error(
            "Exactly 4 positional arguments required: "
            "dataset model peft_config checkpoint_name"
        )

    FOLDER_PATHS = ["src.datasets.", "src.models.", "src.peft_configs."]

    dataset_class = import_class_from_string(FOLDER_PATHS[0] + args.full_class_names[0])
    model_class   = import_class_from_string(FOLDER_PATHS[1] + args.full_class_names[1])
    peft_class    = import_class_from_string(FOLDER_PATHS[2] + args.full_class_names[2])
    checkpoint_name = args.full_class_names[3]

    if not all([dataset_class, model_class, peft_class]):
        sys.exit(1)

    from src.rewards.ner_reward import ner_reward_func
    from trl import GRPOTrainer, GRPOConfig

    # ------------------------------------------------------------------
    # Load model (optionally warm-started from an SFT LoRA checkpoint)
    # ------------------------------------------------------------------
    model_loader = model_class(
        model_name=args.model_name,
        **({"device": args.device} if args.device else {}),
    )
    model_loader.model = model_loader.load_model(adapter_path=args.sft_checkpoint)
    tokenizer = model_loader.tokenizer

    print(f"Model loaded. SFT adapter: {args.sft_checkpoint or 'none (base model)'}")

    # ------------------------------------------------------------------
    # Load dataset in GRPO format
    # ------------------------------------------------------------------
    dataset_loader = dataset_class(tokenizer=tokenizer)
    data = dataset_loader.load_dataset(format="grpo")
    print(f"Dataset loaded: {data}")

    # ------------------------------------------------------------------
    # Apply a fresh LoRA adapter for GRPO training
    # ------------------------------------------------------------------
    peft_loader = peft_class(model=model_loader.model, lora_preset=args.lora_preset)
    grpo_model = peft_loader.apply_lora()

    # ------------------------------------------------------------------
    # GRPO training
    # ------------------------------------------------------------------
    output_dir = f"./outputs/checkpoints/grpo/{checkpoint_name}"

    training_args = GRPOConfig(
        output_dir=output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        logging_steps=args.logging_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        report_to="tensorboard",
        logging_dir=f"./runs/grpo/{checkpoint_name}",
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        beta=args.beta,
    )

    trainer = GRPOTrainer(
        model=grpo_model,
        args=training_args,
        train_dataset=data["train"],
        processing_class=tokenizer,
        reward_funcs=ner_reward_func,
    )

    print(f"Starting GRPO training. Output: {output_dir}")
    trainer.train()
    print("Training complete.")


if __name__ == "__main__":
    main()
