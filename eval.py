import evaluate
from tqdm import tqdm

# ---------- Example data ----------
# Replace these with outputs from your RAG pipeline
queries = [
    "My hip has been sore and stiff. What surgery might I need?",
    "What are the treatments for ACL tear?",
    "How long does knee replacement recovery take?"
]

# Model predictions (from FLAN-T5)
predictions = [
    "Hip pain can result from arthritis or injury. Surgery options include arthroscopy or total hip replacement.",
    "ACL tear treatment includes physiotherapy, ACL reconstruction surgery, or conservative management depending on severity.",
    "Recovery from knee replacement usually takes 6 to 12 weeks, with physical therapy recommended."
]

# Reference / ground truth answers
references = [
    "Patients with hip stiffness may need hip arthroscopy or total hip replacement depending on severity.",
    "ACL injuries are treated with physiotherapy or ACL reconstruction surgery as needed.",
    "Knee replacement recovery typically lasts 6 to 12 weeks with rehabilitation."
]

# Simulated retrieved doc IDs for evaluation (for RAG retrieval)
retrieved_docs = [
    [1, 2, 3],  # top-k retrieved for query 1
    [4, 5, 6],
    [7, 8, 9]
]

# Gold doc IDs
gold_docs = [
    [2],
    [5],
    [8]
]

# ---------- SURFACE METRICS ----------
# BLEU 
bleu = evaluate.load("bleu")
bleu_result = bleu.compute(
    predictions=[p.lower() for p in predictions],
    references=[[r.lower()] for r in references]
)
print(f"BLEU score: {bleu_result['bleu']*100:.2f}")

# ROUGE
rouge = evaluate.load("rouge")
rouge_result = rouge.compute(
    predictions=[p.lower() for p in predictions],
    references=[r.lower() for r in references]
)
print("ROUGE scores:")
for k, v in rouge_result.items():
    print(f"  {k}: {v*100:.2f}")

# ---------- SEMANTIC METRICS ----------
# BERTScore
bertscore = evaluate.load("bertscore")
bert_result = bertscore.compute(
    predictions=predictions,
    references=references,
    lang="en"
)
bert_f1 = sum(bert_result['f1']) / len(bert_result['f1'])
print(f"BERTScore F1: {bert_f1:.4f}")



# ---------- RETRIEVAL METRIC ----------
def recall_at_k(retrieved, gold, k=3):
    total = len(gold)
    correct = 0
    for r_docs, g_docs in zip(retrieved, gold):
        top_k = r_docs[:k]
        if any(doc in top_k for doc in g_docs):
            correct += 1
    return correct / total

recall3 = recall_at_k(retrieved_docs, gold_docs, k=3)
print(f"Retrieval Recall@3: {recall3*100:.2f}%")

# ---------- SAMPLE OUTPUTS ----------
print("\nSample predictions:\n")
for i in range(len(predictions)):
    print(f"Query: {queries[i]}")
    print(f"Reference: {references[i]}")
    print(f"Prediction: {predictions[i]}")
    print("-"*50)
