import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch

# Load the RoBERTa model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# Function to calculate the cosine similarity between two vectors
def cosine_similarity(a, b):
    a = a.squeeze()
    b = b.squeeze()
    return a @ b / (a.norm() * b.norm())

# Load the Excel file
df = pd.read_excel('excelAI.xlsx')

df['j'] = ''

# Get the questions from the first column
questions = df.iloc[:, 0].tolist()

# Calculate RoBERTa embeddings for each question
embeddings = []
for question in questions:
    inputs = tokenizer(question, return_tensors='pt', truncation=True, padding=True)
    outputs = model(**inputs)
    embeddings.append(outputs.last_hidden_state.mean(dim=1).detach())

# Compare each question with the rest
for i in range(len(questions)):
    for j in range(i+1, len(questions)):
        # Calculate the similarity score
        score = cosine_similarity(embeddings[i], embeddings[j])
        print(f'Similarity score between question {i+1} and question {j+1}: {score}')


    # If the score is high enough, they are considered similar
        if score > 0.9:  # You can adjust this threshold
            # Add a link to the similar question (modify this according to your needs)
            df.loc[i, 'j'] += ' -> similar to row ' + str(j+1)

# Save the modified DataFrame back to Excel
df.to_excel('modified_file_roberta.xlsx', index=False)