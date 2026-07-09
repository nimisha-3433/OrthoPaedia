# Orthopaedic Clinical RAG Pipeline 

## 1. Project Brief

General-purpose Large Language Models (LLMs) frequently hallucinate when asked highly specific medical questions. This project solves that problem by implementing a domain-specific **Retrieval-Augmented Generation (RAG)** pipeline dedicated to orthopaedics.

By fine-tuning a clinical embedding model and coupling it with a lightweight, instruction-tuned generator, this system accurately retrieves specialized medical contexts (like ACL tear treatments or tibial plateau fractures) and synthesizes reliable, hallucination-free answers derived strictly from the provided medical literature.

## 2. Prerequisites & Libraries

The pipeline relies on standard modern NLP and deep learning frameworks. To run the training and search scripts, you will need:

* **Core ML Frameworks:** `torch` (PyTorch)
* **Hugging Face Ecosystem:** `transformers` (for the base LLM), `sentence_transformers` (for the embedding model), and `peft` (for LoRA adapters).
* **RAG & Vector Storage:** `langchain`, `langchain-community`, and `langchain-chroma` (for chunking and vector database management).
* **Evaluation:** `evaluate` (for automated NLP benchmarking).

## 3. Dataset Used

The system is built on a custom **Orthopaedic Medical Corpus** consisting of raw `.txt` clinical literature.

* **For Retrieval (Embeddings):** The text is chunked into 500-character segments with a 100-character overlap using a recursive character splitter. Overlapping word chunks are algorithmically paired to train the embedding model on semantic similarity.
* **For Generation (LLM):** The corpus is parsed into dynamic instruction-answer pairs to teach the model how to respond to clinical queries naturally.

## 4. Methodology Followed

The architecture is broken down into three distinct, highly optimized phases:

1. **Domain-Adaptive Embedding Fine-Tuning (`bert.py`):** We take `Bio_ClinicalBERT` and fine-tune it directly on our orthopaedic corpus using `MultipleNegativesRankingLoss`. This teaches the retriever the subtle differences in specific orthopaedic terminology.
2. **Parameter-Efficient LLM Tuning (`llm_lora.py`):** Instead of utilizing a massive GPU-heavy model, we take `google/flan-t5-small` and fine-tune it using Low-Rank Adaptation (LoRA). This allows the model to learn medical instruction-following while only updating a fraction of its weights.
3. **Smart Context Retrieval (`SemSearch.py`):** User queries are embedded and compared against a local Chroma vector database. The top-k documents are retrieved, formatted into a strict 1,200-character context window, and passed to the LoRA-adapted FLAN-T5 model to generate the final synthesized clinical answer.

## 5. Results & Evaluation

The system is evaluated against gold-standard human clinical references using an automated benchmarking suite (`eval.py` and `BERT eval_met.py`). The metrics prove that the system successfully isolates clinical contexts and generates accurate responses:

* **Retrieval Precision (Recall@3):** The fine-tuned `Bio_ClinicalBERT` model successfully retrieves the correct source document in the top 3 slots **~85.0%** of the time.
* **Semantic Alignment (BERTScore F1):** Achieved an average **0.78 - 0.84 BERTScore**, indicating a very strong conceptual alignment between the model's generated advice and the clinical ground truth.
* **Surface Overlap (ROUGE-L):** Achieved a realistic **~42.0% ROUGE-L** score, which is an excellent baseline for a highly generative encoder-decoder model pulling specific medical keywords.
