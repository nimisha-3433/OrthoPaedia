
# Orthopaedic LLM Instruction Tuning
# Using flan-t5-small + LoRA


import os
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
import torch

# CONFIGURATION

BASE_MODEL = "google/flan-t5-small"  # Encoder-decoder model
CORPUS_DIR = r"D:\Sem III projects\NLP\orthopaedic corpus"   # Folder containing .txt files
SAVE_DIR = r"D:\Sem III projects\NLP\flan-t5-ortho-finetuned"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# STEP 1: LOAD CORPUS

def load_corpus(corpus_dir):
    """
    Reads all .txt files and returns a list of dicts: {'instruction': ..., 'output': ...}
    """
    data = []
    for fname in os.listdir(corpus_dir):
        if fname.endswith(".txt"):
            with open(os.path.join(corpus_dir, fname), "r", encoding="utf-8") as f:
                text = f.read().strip()
                if text:
                    sentences = text.split(". ")
                    instruction = sentences[0] + "." if sentences else text
                    output = ". ".join(sentences[1:]) if len(sentences) > 1 else instruction
                    data.append({"instruction": instruction, "output": output})
    return data

corpus_data = load_corpus(CORPUS_DIR)
print(f" Loaded {len(corpus_data)} examples from {CORPUS_DIR}")


# STEP 2: PREPARE HUGGINGFACE DATASET

dataset = Dataset.from_list(corpus_data)

# STEP 3: LOAD TOKENIZER & MODEL

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForSeq2SeqLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32
)

# STEP 4: APPLY LoRA FOR EFFICIENT FINE-TUNING

peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q", "v"]
)

model = get_peft_model(model, peft_config)
print(" LoRA adapters applied to the model.")


# STEP 5: TOKENIZATION 

def tokenize_function(batch):
    prompts = [f"Instruction: {inst}\nAnswer:" for inst in batch["instruction"]]
    outputs = batch["output"]

    model_inputs = tokenizer(
        prompts,
        truncation=True,
        max_length=128,
        padding="max_length",
    )

    labels = tokenizer(
        outputs,
        truncation=True,
        max_length=128,
        padding="max_length",
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    batch_size=4, 
    remove_columns=dataset.column_names,
)

print(f" Tokenization complete. Example batch keys: {list(tokenized_dataset.features.keys())}")


# STEP 6: TRAINING SETUP

training_args = TrainingArguments(
    output_dir=SAVE_DIR,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    learning_rate=5e-4,
    fp16=True if DEVICE == "cuda" else False,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    remove_unused_columns=False,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)


# STEP 7: TRAIN
print("Starting fine-tuning...")
trainer.train()


# STEP 8: SAVE MODEL
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)
print(f"Fine-tuned model saved at {SAVE_DIR}")
