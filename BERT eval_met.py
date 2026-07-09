
from sentence_transformers import SentenceTransformer, evaluation
import numpy as np

# Load your fine-tuned model
model_path = r"D:\Sem III projects\NLP\bioclinicalbert-ortho-finetuned"
model = SentenceTransformer(model_path)

# Create evaluation dataset (you can use your corpus or small labeled pairs)
sentences1 = [
    "Hip replacement surgery relieves arthritis pain.",
    "ACL tear is treated with ligament reconstruction.",
    "Fracture healing is supported by immobilization."
]

sentences2 = [
    "Hip arthroplasty is performed to reduce joint pain.",
    "ACL reconstruction surgery repairs torn ligaments.",
    "Casts help bones heal properly."
]

# True similarities (1 = semantically similar, 0 = dissimilar)
scores = [1, 1, 1]

# Example of adding dissimilar ones for balance
sentences1 += ["Osteoporosis weakens bones.", "Arthritis causes joint stiffness."]
sentences2 += ["MRI is a brain scan.", "Lungs help with breathing."]
scores += [0, 0]

# Use STS (semantic textual similarity) evaluator
evaluator = evaluation.EmbeddingSimilarityEvaluator(sentences1, sentences2, scores)
result = evaluator(model)

print(f"Spearman correlation between predicted and true similarity:", result)
