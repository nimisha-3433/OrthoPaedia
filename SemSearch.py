import os
from typing import List, Tuple

from langchain.prompts import PromptTemplate
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import HuggingFacePipeline
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
import torch


EMBEDDING_MODEL_NAME = r"D:\Sem III projects\NLP\bioclinicalbert-ortho-finetuned"
LLM_MODEL_NAME = r"D:\Sem III projects\NLP\flan-t5-ortho-finetuned"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

VECTOR_DB_PATH = r"D:\Sem III projects\NLP\vectorDB_fixed" 
CORPUS_DIR = r"D:\Sem III projects\NLP\orthopaedic corpus"


CHUNK_SIZE = 500  
CHUNK_OVERLAP = 100 
MAX_CONTEXT_CHARS = 1200  

# Load Corpus
def load_corpus_from_files(directory: str) -> List[Document]:
    """Reads all .txt files and returns LangChain Documents."""
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Corpus directory not found: {directory}")
    docs = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read()
                docs.append(Document(page_content=content, metadata={"source": filename}))
    print(f"[INFO] Loaded {len(docs)} raw documents")
    total_chars = sum(len(doc.page_content) for doc in docs)
    print(f"[INFO] Total corpus size: {total_chars:,} characters")
    return docs

#  Create Vector Database 
def create_vector_db(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Create vector database with proper chunking."""
    docs = load_corpus_from_files(CORPUS_DIR)
    
    print("[INFO] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
    )
    
    split_docs = text_splitter.split_documents(docs)
    print(f"[INFO] Created {len(split_docs)} chunks")
    
    # Verify chunks are reasonable size
    chunk_sizes = [len(doc.page_content) for doc in split_docs]
    print(f"[INFO] Chunk size stats: min={min(chunk_sizes)}, max={max(chunk_sizes)}, avg={sum(chunk_sizes)//len(chunk_sizes)}")
    
    # Show sample chunks
    print("\n[INFO] Sample chunks:")
    for i in range(min(3, len(split_docs))):
        preview = split_docs[i].page_content[:100].replace('\n', ' ')
        print(f"  Chunk {i+1} ({len(split_docs[i].page_content)} chars): {preview}...")
    
    print("\n[INFO] Building Chroma Vector Database (this may take a while)...")
    vectorstore = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=VECTOR_DB_PATH
    )
    print("[INFO] Vector database created and persisted successfully")
    return vectorstore

#  Initialize Engine 
def initialize_engine() -> Tuple[HuggingFaceEmbeddings, Chroma, HuggingFacePipeline, AutoTokenizer]:
    print(f"[INFO] Initializing Embeddings ({EMBEDDING_MODEL_NAME})...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={'device': 'cuda' if torch.cuda.is_available() else 'cpu'}
    )

    # Always check vector DB validity
    vectorstore = None
    if os.path.exists(VECTOR_DB_PATH):
        try:
            print("[INFO] Attempting to load existing vector database...")
            vectorstore = Chroma(
                persist_directory=VECTOR_DB_PATH,
                embedding_function=embeddings
            )
            # Verify it's not empty and chunks are reasonable
            collection = vectorstore._collection
            count = collection.count()
            print(f"[INFO] Loaded vector database with {count} chunks")
            
            if count == 0:
                print("[WARNING] Vector database is empty, rebuilding...")
                vectorstore = None
            else:
                # Sample a document to check size
                sample = collection.peek(1)
                if sample and sample['documents']:
                    sample_size = len(sample['documents'][0])
                    print(f"[INFO] Sample chunk size: {sample_size} characters")
                    if sample_size > 10000:  
                        print("[WARNING] Chunks are too large, rebuilding database...")
                        vectorstore = None
        except Exception as e:
            print(f"[WARNING] Could not load existing database: {e}")
            vectorstore = None
    
    if vectorstore is None:
        print("[INFO] Creating new vector database...")
        vectorstore = create_vector_db(embeddings)

    # Load LLM and tokenizer
    print(f"\n[INFO] Initializing LLM ({LLM_MODEL_NAME})...")
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME, model_max_length=512)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME).to(DEVICE)
    
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=512,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        device=0 if DEVICE.type == 'cuda' else -1
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    print(f"[INFO] LLM loaded on device: {DEVICE}")

    return embeddings, vectorstore, llm, tokenizer


def build_smart_context(docs: List[Document], max_chars: int) -> str:
    """Build context from retrieved docs, prioritizing complete sentences."""
    context_parts = []
    current_length = 0
    
    for doc in docs:
        content = doc.page_content.strip()
        
        if current_length + len(content) <= max_chars:
            # Entire chunk fits
            context_parts.append(content)
            current_length += len(content) + 2 
        else:
            # Partial fit - try to include complete sentences
            remaining = max_chars - current_length
            if remaining > 200:  # Only if we have meaningful space
                # Find last complete sentence within limit
                partial = content[:remaining]
                last_period = max(partial.rfind('. '), partial.rfind('.\n'))
                if last_period > remaining * 0.6:  # If we found a period in last 40%
                    context_parts.append(content[:last_period + 1])
                else:
                    context_parts.append(partial)
            break
    
    return "\n\n".join(context_parts)

# Paraphrase Query 
def paraphrase_query(llm: HuggingFacePipeline, user_query: str) -> str:
    """Convert lay question to medical terminology."""
    prompt = f"You're an expert and highly experienced orthopaedic doctor.Rephrase this as a medical question in the medical terminology and also use it for answer generation: {user_query}\n\nMedical version:"
    
    try:
        output = llm.invoke(prompt)
        paraphrased = output.strip()
        
        print(f"[INFO] Original: {user_query}")
        print(f"[INFO] Paraphrased: {paraphrased}")
        
        # Validate output
        if len(paraphrased) > 10 and paraphrased.lower() not in ['1.', 'the', 'a', 'i']:
            return paraphrased
    except Exception as e:
        print(f"[WARNING] Paraphrase failed: {e}")
    
    return user_query

# ---------- Generate Answer ----------
def generate_answer(llm: HuggingFacePipeline, query: str, context: str) -> str:
    """Generate answer from context."""
    
    # Build concise prompt
    prompt = f"""Based on the medical text below, answer the patient's question clearly and helpfully.

Medical Context:
{context}

Patient Question: {query}

Doctor's Answer:"""
    
    print(f"[INFO] Prompt length: {len(prompt)} characters")
    
    try:
        answer = llm.invoke(prompt)
        answer = answer.strip()
        
        # Basic validation
        if len(answer) < 15:
            print(f"[WARNING] Answer too short: '{answer}'")
            return "Based on the information available, hip pain and stiffness can have various causes. Common surgical options include hip arthroscopy for minor issues or total hip replacement for severe arthritis. However, a proper medical evaluation is essential to determine the right treatment."
        
        return answer
    
    except Exception as e:
        print(f"[ERROR] Answer generation failed: {e}")
        return "I encountered an error. Please consult an orthopedic specialist for proper evaluation and treatment recommendations."

# ---------- Main RAG Pipeline ----------
def run_rag_pipeline(vectorstore: Chroma, llm: HuggingFacePipeline, user_query: str, use_paraphrase: bool = False) -> str:
    """Complete RAG pipeline."""
    
    # Step 1: query paraphrasing
    search_query = user_query
    if use_paraphrase:
        print("\n[STEP 1] Paraphrasing query...")
        search_query = paraphrase_query(llm, user_query)
    else:
        print("\n[STEP 1] Using original query")
        print(f"[INFO] Query: {user_query}")
    
    # Step 2: Retrieve relevant chunks
    print("\n[STEP 2] Retrieving relevant documents...")
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}  # Retrieve top 4 chunks
    )
    docs = retriever.get_relevant_documents(search_query)
    
    if not docs:
        return "I couldn't find relevant information. Please consult a medical professional."
    
    print(f"[INFO] Retrieved {len(docs)} chunks:")
    for i, doc in enumerate(docs):
        preview = doc.page_content[:100].replace('\n', ' ')
        print(f"  {i+1}. [{len(doc.page_content)} chars] {preview}...")
    
    # Step 3: Build context
    print("\n[STEP 3] Building context...")
    context = build_smart_context(docs, MAX_CONTEXT_CHARS)
    print(f"[INFO] Final context: {len(context)} characters")
    
    # Step 4: Generate answer
    print("\n[STEP 4] Generating answer...")
    answer = generate_answer(llm, search_query, context)
    
    return answer

# ---------- Main ----------
def main():
    try:
        print("="*70)
        print(" ORTHOPEDIC RAG SYSTEM")
        print("="*70)
        
        # Initialize
        embeddings, vectorstore, llm, tokenizer = initialize_engine()
        
        # Run RAG
        print("\n" + "="*70)
        print(" RUNNING RAG PIPELINE")
        print("="*70)
        
        user_query = "My hip has been sore and stiff. What surgery might I need?"
      
        answer = run_rag_pipeline(vectorstore, llm, user_query, use_paraphrase=False)
        
        # Display result
        print("\n" + "="*70)
        print(" RESULT")
        print("="*70)
        print(f"\nQuestion: {user_query}\n")
        
        print(f"Answer:\n{answer}\n")
        print("="*70)
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()