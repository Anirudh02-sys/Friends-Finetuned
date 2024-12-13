import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
import wandb
from dotenv import load_dotenv
import os
import random


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

SEED = 595
set_seed(SEED)

# Detect device
device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Load WandB credentials
load_dotenv()
api_key = os.getenv('WANDB_API_KEY')
wandb.login(key=api_key)

# Initialize WandB
# CHANGE STUFF HERE !!
run = wandb.init(
    project="Finetuning Friends-Mistral",
    job_type="training",
    name="chandler_finetuning_800",
    anonymous="allow"
)

# Load dataset and select a subset
dataset = load_dataset("json", data_files="friends_jsonl/chandler_bing_dialogues.jsonl") # CHANGE stuff here
dataset = dataset['train']  # Use the 'train' split
dataset = dataset.select(range(800))  # Reduced to 1000 samples

# Split dataset into train and evaluation
train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split['train']
eval_dataset = train_test_split['test']

# Load tokenizer
base_model = "mistralai/Mistral-7B-v0.3"
tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
tokenizer.padding_side = 'right'
tokenizer.pad_token = tokenizer.eos_token

# Preprocessing function with system prompt
def preprocess_function(examples):
    concatenated_messages = []
    for messages in examples.get('messages', []):
        # Add a system prompt to guide character persona
        # CHANGE STUFF HERE
        conversation = "[SYSTEM] You are Chandler Bing from the TV show 'Friends'. You are known for your sharp, acerbic sense of humor and deadpan sarcasm. You often use wit to compensate for your insecurities, especially your fear of commitment. You are financially successful, support your best friend Joey, and are deeply in love with Monica, with whom you share a stable relationship. Despite your humor, you have a caring and loyal nature. Respond to conversations with your characteristic wit, humor, and occasional self-deprecating remarks.\n"
        for message in messages:
            if message.get('role') == 'user':
                conversation += f"[INST] {message.get('content', '')} [/INST] "
            elif message.get('role') == 'assistant':
                conversation += f"{message.get('content', '')} "
        concatenated_messages.append(conversation.strip())
    
    # Tokenize and truncate messages
    max_length = min(tokenizer.model_max_length, 512)  # Reduced max_length
    return tokenizer(
        concatenated_messages,
        truncation=True,
        padding='max_length',
        max_length=max_length
    )

# Apply preprocessing to train and eval datasets
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

# Add labels for causal language modeling
def add_labels(example):
    example['labels'] = example['input_ids'].copy()
    return example

train_dataset = train_dataset.map(add_labels)
eval_dataset = eval_dataset.map(add_labels)

# Load the model with 4-bit precision
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

# Prepare the model for LoRA fine-tuning
model = prepare_model_for_kbit_training(model)

# Configure LoRA
lora_config = LoraConfig(
    r=4,  # Smaller rank for faster training
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./friends_mistral", # CHANGE THIS !!!
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,  # Optimized for faster training
    evaluation_strategy="steps",
    eval_steps=100,  # Evaluate every 100 steps
    learning_rate=1e-4,  # Reduced for stable training
    lr_scheduler_type="cosine",  # Smooth learning rate decay
    warmup_steps=100,  # Stabilize early training
    num_train_epochs=2,  # Fewer epochs for faster convergence
    save_strategy="steps",
    save_steps=200,
    logging_steps=10,
    report_to="wandb",
    run_name="mistral-7b-finetuning-chandler",
    fp16=True,
    max_grad_norm=1.0,  # Clip gradients for stability
)

# Data collator with dynamic padding
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # No masked language modeling for causal tasks
    pad_to_multiple_of=8,  # Dynamic padding for memory efficiency
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    data_collator=data_collator,
   
)

# Start training
trainer.train()

# Save the LoRA adapter and tokenizer
model.save_pretrained(training_args.output_dir)
tokenizer.save_pretrained(training_args.output_dir)
