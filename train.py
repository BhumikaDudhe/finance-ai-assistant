from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
import torch

# -----------------------------------
# MODEL NAME
# -----------------------------------

model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# -----------------------------------
# LOAD TOKENIZER
# -----------------------------------

tokenizer = AutoTokenizer.from_pretrained(model_name)

tokenizer.pad_token = tokenizer.eos_token

# -----------------------------------
# LOAD MODEL
# -----------------------------------

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float32
)

# -----------------------------------
# LoRA CONFIG
# -----------------------------------

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# -----------------------------------
# LOAD DATASET
# -----------------------------------

dataset = load_dataset(
    "json",
    data_files="dataset/finance_data.json"
)

# -----------------------------------
# FORMAT DATA
# -----------------------------------

def format_data(example):
    text = f"""
    ### Instruction:
    {example['instruction']}

    ### Response:
    {example['response']}
    """

    tokens = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=256
    )

    tokens["labels"] = tokens["input_ids"].copy()

    return tokens

tokenized_dataset = dataset.map(format_data)

# -----------------------------------
# TRAINING ARGUMENTS
# -----------------------------------

training_args = TrainingArguments(
    output_dir="./saved_model",
    per_device_train_batch_size=1,
    num_train_epochs=5,
    logging_steps=1,
    save_steps=10,
    learning_rate=2e-4,
    fp16=False,
    report_to="none"
)

# -----------------------------------
# TRAINER
# -----------------------------------

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
)

# -----------------------------------
# START TRAINING
# -----------------------------------

print("Training started...")

trainer.train()

# -----------------------------------
# SAVE MODEL
# -----------------------------------

model.save_pretrained("./saved_model")

tokenizer.save_pretrained("./saved_model")

print("Model training completed!")
print("Model saved in saved_model folder")