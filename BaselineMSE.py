import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification, DistilBertTokenizer, DistilBertForSequenceClassification
from sklearn.metrics import mean_squared_error
from tqdm import tqdm

# Load the dataset
df = pd.read_csv('soft_targets.csv')

# Create the BERT input sequence from adjective and noun separated by SEP
def create_input_sequence(row):
    return f"[CLS] {row['Adjective']} [SEP] {row['Noun']} [SEP]"

df['input_sequence'] = df.apply(create_input_sequence, axis=1)

# Load the tokenizers and models
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
distilbert_tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
teacher_model = BertForSequenceClassification.from_pretrained('./finetuned_bert_for_similarity')
student_model = DistilBertForSequenceClassification.from_pretrained('./finetuned_student_model')

teacher_model.eval()
student_model.eval()

teacher_predictions = []
student_predictions = []

# Generate predictions with progress bar
with torch.no_grad():
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
        inputs_teacher = bert_tokenizer(row['input_sequence'], return_tensors='pt', padding=True, truncation=True, max_length=128)
        inputs_student = distilbert_tokenizer(row['input_sequence'], return_tensors='pt', padding=True, truncation=True, max_length=128)

        outputs_teacher = teacher_model(**inputs_teacher)
        outputs_student = student_model(**inputs_student)

        teacher_score = outputs_teacher.logits.item()
        student_score = outputs_student.logits.item()

        teacher_predictions.append(teacher_score)
        student_predictions.append(student_score)

df['Teacher_Predictions'] = teacher_predictions
df['Student_Predictions'] = student_predictions

# Calculate MSE between teacher predictions and student predictions
mse = mean_squared_error(df['Teacher_Predictions'], df['Student_Predictions'])

# Print evaluation summary
print("\nEvaluation Summary")
print("==================")
print(f"Number of Samples: {len(df)}")
print(f"Mean Squared Error (MSE) between Teacher and Student Predictions: {mse}")

# Optional: Save the evaluation results to a file
df.to_csv('evaluation_results.csv', index=False)
print("\nEvaluation results have been saved to 'evaluation_results.csv'.")

# Evaluation Summary
# ==================
# Number of Samples: 134700
# Mean Squared Error (MSE) between Teacher and Student Predictions: 6.320421138100353e-05
#
# Evaluation results have been saved to 'evaluation_results.csv'.

