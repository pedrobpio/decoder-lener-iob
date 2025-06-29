
# main_script_multi_module.py
from accelerate import Accelerator
import sys
import importlib
import argparse
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
import torch
sys.path.append('..')

def import_class_from_string(full_class_string):
    """
    Dynamically imports a class using a string like 'module_name.ClassName'.
    Returns the class object or None if import fails.
    """
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
        print(f"An unexpected error occurred during import of {full_class_string}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Dynamically import classes using 'module.Class' format.")
    parser.add_argument('full_class_names',
                        metavar='module.ClassName',
                        type=str,
                        nargs='+',
                        help='Fully qualified names of classes to import (e.g., my_classes.Dog)')

    args = parser.parse_args()
    print(args.full_class_names)
    print(sys.path)
    imported_classes = {}

    if len(args.full_class_names) != 4:
        print("Make sure you passed 4 arguments in the following order: a dataset, a model, a peft_config, and the checkpoint reference.")
        sys.exit(1)

    # Define the module where the classes are expected to be found
    FOLDER_PATHS = ["src.datasets.", "src.models.", "src.peft_configs."]

    for index, class_name in enumerate(args.full_class_names):
        if index == 3:
            checkpoint_reference = class_name
            print(f"Checkpoint reference: {checkpoint_reference}")
            break
        full_name = FOLDER_PATHS[index]+class_name
        class_obj = import_class_from_string(full_name)
        if class_obj:
            # Store using the full name or just the class name as key
            imported_classes[full_name] = class_obj
            if index == 0:
                Dataset_loader = class_obj
            if index == 1:
                Model_loader = class_obj
            if index == 2:
                Peft_configs_loader = class_obj

    # Instanciating the classes
    # loading the model
    model_loader = Model_loader()
    model_loader.load_tokenizer()
    model_loader.load_model()
    # Save the tokenizer in a variable
    tokenizer = model_loader.tokenizer

    # loading the dataset
    dataset_loader = Dataset_loader(tokenizer = tokenizer)
    data = dataset_loader.load_dataset()

    peft_configs_loader = Peft_configs_loader(model_loader.model, r=8, lora_alpha=16, lora_dropout=0.1, bias="none")

    lora_model = peft_configs_loader.apply_lora()

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    # Training args
    training_args = TrainingArguments(
        output_dir=f"./outputs/checkpoints/{checkpoint_reference}",
        num_train_epochs=3,
        per_device_train_batch_size=16,
        gradient_accumulation_steps=4,
        logging_steps=20,
        save_strategy="epoch",
        eval_strategy="no",
        learning_rate=2e-4,
        warmup_steps=10,
        fp16=torch.cuda.is_available(),
        report_to="none"
    )

    accelerator = Accelerator()
    
    # Trainer
    trainer = accelerator.prepare(Trainer(
        model=lora_model,
        args=training_args,
        train_dataset=data['train'],
        tokenizer=tokenizer,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    ))

    trainer.train()

if __name__ == "__main__":
    main()