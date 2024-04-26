import pandas as pd
from sentence_transformers import SentenceTransformer, util

# Load the LaBSE model
labse_model = SentenceTransformer('stsb-roberta-base-v2')

# Function to calculate the cosine similarity between two vectors
def cosine_similarity(a, b):
    a = a.squeeze()
    b = b.squeeze()
    return a @ b / (a.norm() * b.norm())

# Load the Excel file
df = pd.read_excel('excelAI.xlsx')

# Create the 'j' column
df['j'] = ''

# Get the questions from the first column
questions = df.iloc[:, 0].tolist()

# Calculate LaBSE embeddings for each question
embeddings = labse_model.encode(questions, convert_to_tensor=True)

# Compare each question with the rest
for i in range(len(questions)):
    for j in range(i+1, len(questions)):
        print(f'the questions are ')
        # Calculate the similarity score
        score = cosine_similarity(embeddings[i], embeddings[j])
        print(f'Similarity score between question {i+1} and question {j+1}: {score}')


    # If the score is high enough, they are considered similar
        if score > 0.9:  # You can adjust this threshold
            # Add a link to the similar question (modify this according to your needs)
            df.loc[i, 'j'] += ' -> similar to row ' + str(j+1)

# Save the modified DataFrame back to Excel
df.to_excel('modified_file_labse.xlsx', index=False)
