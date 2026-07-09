# fine_tune_bioclinicalbert


import os
import re
import traceback
from sentence_transformers import SentenceTransformer, InputExample, models, losses
from torch.utils.data import DataLoader


BASE_MODEL = "emilyalsentzer/Bio_ClinicalBERT"
CORPUS_DIR = r"D:\Sem III projects\NLP\orthopaedic corpus"   # folder of .txt files
SAVE_DIR = r"D:\Sem III projects\NLP\bioclinicalbert-ortho-finetuned"
EPOCHS = 3
BATCH_SIZE = 8                 
MAX_PAIRS_PER_FILE = 500       # cap pairs per file
MIN_TOTAL_PAIRS = 8            # minimal pairs required to start training
CHUNK_WORDS = 120              
CHUNK_STEP = 60                


def split_into_sentences(text):
    """Simple sentence splitter — returns sentences of length >= 5 chars."""
    sents = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 5]

def chunk_text_words(text, chunk_words=CHUNK_WORDS, step=CHUNK_STEP):
    """Create overlapping word chunks from text."""
    words = text.split()
    if len(words) <= chunk_words:
        return [" ".join(words)]
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+chunk_words]
        if not chunk:
            break
        chunks.append(" ".join(chunk))
        i += step
        if len(chunks) >= 1000:
            break
    return chunks

def create_pairs_from_text(text, max_pairs=MAX_PAIRS_PER_FILE):
    """
    Create positive pairs from a single document:
      - adjacent sentences pairs
      - plus some combined-sentence positives
      - if not enough, fallback to overlapping word-chunks
    """
    examples = []
    sents = split_into_sentences(text)
    # adjacent sentence pairs
    for i in range(len(sents) - 1):
        examples.append(InputExample(texts=[sents[i], sents[i+1]]))
        if len(examples) >= max_pairs:
            return examples
    # add slightly longer positives (i -> i+1+i+2)
    for i in range(len(sents) - 2):
        p = " ".join(sents[i+1:i+3])
        examples.append(InputExample(texts=[sents[i], p]))
        if len(examples) >= max_pairs:
            return examples
    # fallback: chunk into overlapping windows of words
    if len(examples) < 4:
        chunks = chunk_text_words(text)
        for i in range(len(chunks) - 1):
            examples.append(InputExample(texts=[chunks[i], chunks[i+1]]))
            if len(examples) >= max_pairs:
                break
    return examples

# ---------- MAIN ----------
def main():
    try:
        if not os.path.isdir(CORPUS_DIR):
            raise FileNotFoundError(f"Corpus directory not found: {CORPUS_DIR}")

        files = [f for f in os.listdir(CORPUS_DIR) if f.lower().endswith(".txt")]
        if not files:
            raise FileNotFoundError(f"No .txt files found in {CORPUS_DIR}")

        all_examples = []
        for fname in files:
            fp = os.path.join(CORPUS_DIR, fname)
            try:
                with open(fp, "r", encoding="utf-8", errors="ignore") as fh:
                    text = fh.read().strip()
            except Exception as e:
                print(f"Warning: could not read {fname}: {e}")
                continue
            if not text:
                continue
            pairs = create_pairs_from_text(text, max_pairs=MAX_PAIRS_PER_FILE)
            if pairs:
                all_examples.extend(pairs)
            print(f"Processed {fname}: sentences={len(split_into_sentences(text))}, pairs_created={len(pairs)}")

        print(f"\nTotal positive pairs created: {len(all_examples)}")

        if len(all_examples) < MIN_TOTAL_PAIRS:
            raise ValueError(
                f"Too few training pairs ({len(all_examples)}). "
                "Add more .txt files or adjust CHUNK_WORDS/CHUNK_STEP to create more pairs."
            )


        # Build SentenceTransformer model 
        print("\nBuilding SentenceTransformer model...")
        word_emb = models.Transformer(BASE_MODEL, max_seq_length=256)
        pool = models.Pooling(word_emb.get_word_embedding_dimension(), pooling_mode_mean_tokens=True)
        model = SentenceTransformer(modules=[word_emb, pool])

        # DataLoader and loss
        train_dataloader = DataLoader(all_examples, shuffle=True, batch_size=BATCH_SIZE)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        
        steps_per_epoch = max(1, len(train_dataloader))
        warmup_steps = int(0.1 * steps_per_epoch * EPOCHS)

        os.makedirs(SAVE_DIR, exist_ok=True)
        print(f"\nTraining: epochs={EPOCHS}, batch_size={BATCH_SIZE}, warmup_steps={warmup_steps}")

        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=EPOCHS,
            warmup_steps=warmup_steps,
            output_path=SAVE_DIR
        )

        print(f"\nSaved fine-tuned model to: {SAVE_DIR}")

        # Quick usage check
        loaded = SentenceTransformer(SAVE_DIR)
        q = "What are common treatments for ACL tear?"
        emb = loaded.encode([q])
        print("Embedding shape:", emb.shape)

    except Exception as e:
        print("\nERROR during fine-tuning:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
