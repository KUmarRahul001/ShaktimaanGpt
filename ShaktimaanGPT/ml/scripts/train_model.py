from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, Trainer, TrainingArguments

# Load the dataset
dataset = load_dataset('squad')

# Load a tokenizer and model for question answering
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
model = AutoModelForQuestionAnswering.from_pretrained('bert-base-uncased')

# Define preprocessing function
def preprocess_function(examples):
    return tokenizer(
        examples['question'],
        examples['context'],
        truncation=True,
        padding='max_length',
        max_length=512
    )

# Tokenize the dataset
tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Load the metric for evaluation
metric = load_metric('squad')

# Define a compute_metrics function to evaluate the model's performance
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = tokenizer.convert_ids_to_tokens(predictions)
    return metric.compute(predictions=predictions, references=labels)

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',  # Temporary output directory
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    weight_decay=0.01,
)

# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets['train'],
    eval_dataset=tokenized_datasets['validation'],
    compute_metrics=compute_metrics,
)

# Train the model
trainer.train()

# Save the model and tokenizer to the specified directory
model_save_path = 'E:/shaktimaan-gpt/ShaktimaanGPT/ml/models'
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

print(f"Model and tokenizer saved to {model_save_path}")
