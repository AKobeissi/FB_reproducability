# Using Your Trained Page Scorer

This guide explains how to use your trained page scorer (`finetuned_page_scorer_v2`) in your retrieval pipeline.

## Pipeline Overview

The **Page-Then-Chunk** retrieval pipeline has 5 stages:

```
Query: "What was Apple's revenue in FY2023?"
   ↓
1. SCORE ALL PAGES (using your trained model)
   → Learned embeddings score each PDF page
   ↓
2. RANK & RETRIEVE TOP M PAGES
   → Get top 5 pages (configurable via page_k)
   ↓
3. CHUNK THOSE PAGES
   → Split the 5 pages into smaller chunks (e.g., 1024 chars)
   ↓
4. SCORE & RANK CHUNKS
   → Rank chunks using base embedding model
   ↓
5. GENERATE ANSWER
   → Use top K chunks (e.g., top 3) for LLM generation
```

## Quick Start

### 1. Verify Your Trained Model Exists

```bash
ls -la models/finetuned_page_scorer_v2/
```

You should see:
- `config.json`
- `pytorch_model.bin` (or `model.safetensors`)
- `tokenizer files`

### 2. Run with Trained Scorer

```bash
python example_use_learned_page_scorer.py
```

This will:
- Load your trained page scorer from `models/finetuned_page_scorer_v2`
- Index all PDF pages using the learned embeddings
- Run retrieval on the FinanceBench dataset
- Save results to `outputs/learned_page_scorer/`

### 3. Compare Learned vs Baseline

```bash
python example_use_learned_page_scorer.py --compare
```

This runs both pipelines and shows which pages each retrieves.

## Configuration Parameters

### Page Retrieval (Stage 1 & 2)

```python
page_k = 5  # How many pages to retrieve
```

**Recommended values:**
- `page_k=3`: Fast, less context
- `page_k=5`: **Default**, good balance
- `page_k=10`: More recall, slower

### Chunking (Stage 3)

```python
chunk_size = 1024      # Characters per chunk
chunk_overlap = 100    # Overlap between chunks
```

**Recommended values:**
- Small chunks: `chunk_size=512, overlap=50` (more granular)
- **Default**: `chunk_size=1024, overlap=100` (balanced)
- Large chunks: `chunk_size=2048, overlap=200` (more context)

### Chunk Retrieval (Stage 4 & 5)

```python
top_k = 3  # How many chunks to use for generation
```

**Recommended values:**
- `top_k=1`: Minimal context
- `top_k=3`: **Default**
- `top_k=5`: More context (risk of noise)

## Integration into Your Code

### Option 1: Use the Helper Function

```python
from src.experiments.page_retrieval import run_page_then_chunk
from src.core.rag_experiments import RAGExperiment

# Configure experiment
experiment = RAGExperiment(
    model_name="meta-llama/Llama-3.2-3B-Instruct",
    page_k=5,    # Top M pages
    top_k=3,     # Top K chunks
    chunk_size=1024,
    pdf_dir="pdfs",
)

# Run with learned scorer
results = run_page_then_chunk(
    experiment=experiment,
    data=your_dataset,
    learned_model_path="models/finetuned_page_scorer_v2"  # ← Key parameter!
)
```

### Option 2: Custom Integration

```python
from sentence_transformers import SentenceTransformer
from langchain.embeddings.base import Embeddings

# Load your trained model
class LearnedPageEmbeddings(Embeddings):
    def __init__(self, model_path: str):
        self.model = SentenceTransformer(model_path)
    
    def embed_documents(self, texts: List[str]):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text: str):
        return self.model.encode([text], convert_to_numpy=True)[0].tolist()

# Use it
page_embeddings = LearnedPageEmbeddings("models/finetuned_page_scorer_v2")

# Now use page_embeddings for indexing/retrieving pages
query_embedding = page_embeddings.embed_query("What was revenue?")
page_embeddings_list = page_embeddings.embed_documents(page_texts)
```

## Understanding the Code Flow

### In `page_retrieval.py`

The key section that uses your model:

```python
# Line ~120-145 in run_page_then_chunk()
if learned_model_path:
    logger.info(f"Loading Learned Page Scorer from: {learned_model_path}")
    
    # Wrap SentenceTransformer in LangChain Embeddings interface
    class STEmbeddings(Embeddings):
        def __init__(self, path: str):
            self.model = SentenceTransformer(path)
        
        def embed_documents(self, texts: List[str]):
            return self.model.encode(texts, convert_to_numpy=True).tolist()
        
        def embed_query(self, text: str):
            return self.model.encode([text], convert_to_numpy=True)[0].tolist()
    
    page_embeddings = STEmbeddings(learned_model_path)  # ← Your model!
    collection_suffix = "learned"
else:
    page_embeddings = experiment.embeddings  # ← Baseline model
    collection_suffix = "baseline"
```

### Vector Store Creation

```python
# Line ~155
db_name = f"pages_v12_text_{collection_suffix}"

# Line ~160-165
_, page_vectordb, is_empty_flag = build_chroma_store(
    experiment, 
    db_name, 
    embeddings=page_embeddings,  # ← Uses your learned embeddings
    lazy_load=True
)
```

**Important:** The vector store is named differently for learned vs baseline, so they don't conflict.

### Retrieval

```python
# Line ~280-285
retrieved_pages = page_vectordb.similarity_search(query, k=page_k)
```

This uses your learned embeddings to find the most relevant pages!

## Advanced: Multi-Vector Page Retrieval

For even better recall, use multiple embeddings per page:

```python
from src.experiments.page_retrieval import run_multivec_page_then_chunk

results = run_multivec_page_then_chunk(
    experiment=experiment,
    data=dataset,
    learned_model_path="models/finetuned_page_scorer_v2",
    page_window_size=800,   # Split each page into 800-char windows
    page_window_overlap=100  # Overlap between windows
)
```

**How it works:**
1. Each page → 3-6 overlapping windows
2. Each window gets its own embedding
3. At retrieval: aggregate windows by max score per page
4. Result: Better recall (catches partial matches within pages)

## Evaluation

After running your pipeline:

```bash
# Evaluate retrieval quality
python src/evaluation/evaluate_outputs.py \
    --input outputs/learned_page_scorer/learned_page_scorer_TIMESTAMP.json

# Evaluate answer quality (if you have a judge model)
python src/evaluation/generative_evaluator.py \
    --input outputs/learned_page_scorer/learned_page_scorer_TIMESTAMP.json
```

## Troubleshooting

### Model Not Found

```
❌ Trained model not found at: models/finetuned_page_scorer_v2
```

**Solution:** Train the model first:
```bash
python src/training/train_page_scorer_v2.py
```

### CUDA Out of Memory

**During indexing:**
```python
# Reduce batch size in indexing
# Edit page_retrieval.py line ~220
populate_chroma_store(experiment, page_vectordb, page_docs, db_name, batch_size=32)
```

**During inference:**
```python
# Reduce page_k or top_k
experiment.page_k = 3  # Instead of 5
experiment.top_k = 2   # Instead of 3
```

### Slow Performance

1. **First run is slow** (building index): Normal, it caches the embeddings
2. **Subsequent runs**: Should be fast (uses cached vector store)
3. **To rebuild index**: Delete `vector_stores/chroma/pages_v12_text_learned/`

## File Locations

```
FB_reproducability/
├── models/
│   └── finetuned_page_scorer_v2/    ← Your trained model
│       ├── config.json
│       ├── pytorch_model.bin
│       └── ...
├── src/
│   ├── training/
│   │   └── train_page_scorer_v2.py  ← Training script
│   └── experiments/
│       └── page_retrieval.py        ← Pipeline implementation
├── example_use_learned_page_scorer.py  ← This example
├── vector_stores/
│   └── chroma/
│       ├── pages_v12_text_learned/  ← Indexed with your model
│       └── pages_v12_text_baseline/ ← Indexed with base model
└── outputs/
    └── learned_page_scorer/         ← Results
```

## Next Steps

1. **Run the example:** `python example_use_learned_page_scorer.py`
2. **Compare results:** `python example_use_learned_page_scorer.py --compare`
3. **Tune parameters:** Adjust `page_k`, `chunk_size`, `top_k`
4. **Evaluate:** Use the evaluation scripts
5. **Try multi-vector:** Use `run_multivec_page_then_chunk()` for better recall

## Questions?

- Check [page_retrieval.py](src/experiments/page_retrieval.py) for the full implementation
- Check [train_page_scorer_v2.py](src/training/train_page_scorer_v2.py) to understand the model
- See [LEARNED_PAGE_SCORER_V2.md](LEARNED_PAGE_SCORER_V2.md) for training details
