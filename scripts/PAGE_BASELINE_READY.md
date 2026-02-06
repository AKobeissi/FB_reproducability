# Page-Then-Chunk Baseline - Ready to Run

## ✅ Code Verification

Your baseline page-then-chunk implementation is **correct and ready to run**:

### Key Components:

1. **Page Retrieval** ([src/experiments/page_retrieval.py](../src/experiments/page_retrieval.py))
   - ✓ Uses `experiment.embeddings` (BGE-M3) for baseline
   - ✓ Retrieves top-5 pages by default (`page_k=5`)
   - ✓ Stores pages in Chroma with collection name `pages_v12_text_baseline`

2. **Chunk Scoring** 
   - ✓ Uses `experiment.embeddings` (BGE-M3) for chunks
   - ✓ Chunks retrieved pages on-the-fly (1024 chars, 128 overlap)
   - ✓ Ranks chunks using BGE-M3 embeddings
   - ✓ Uses top-5 chunks for generation (`top_k=5`)

3. **Generation**
   - ✓ Always generates answer using the experiment's LLM (Qwen or Llama)
   - ✓ Compatible with existing evaluators

## 🚀 How to Run

### Option 1: SLURM Batch Job (Recommended)

```bash
sbatch scripts/run_page_baseline.sh
```

This will:
- Request 1 GPU, 32GB RAM, 3 hours
- Run with BGE-M3 embeddings
- Save to `outputs/page_baseline_YYYYMMDD_HHMMSS/`

### Option 2: Interactive with srun

```bash
srun --partition=rali --gres=gpu:1 --mem=32G --time=03:00:00 --pty bash
source venv/bin/activate

python -m FB_reproducability.rag_experiments qwen \
  -e page_baseline \
  --embedding-model "BAAI/bge-m3" \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --top-k 5 \
  --output-dir "outputs/page_baseline_test"
```

### Option 3: Direct Python Call

```bash
source venv/bin/activate

python -m FB_reproducability.rag_experiments qwen \
  -e page_baseline \
  --embedding-model "BAAI/bge-m3" \
  --chunk-size 1024 \
  --chunk-overlap 128 \
  --top-k 5
```

## 📊 Expected Output

The experiment will create:

```
outputs/page_baseline/YYYYMMDD/
├── page_baseline_YYYYMMDD_HHMMSS.json       # Raw results
└── page_baseline_YYYYMMDD_HHMMSS_scored.json  # Evaluated results (after scoring)
```

Each result contains:
- `retrieved_chunks`: Top-5 chunks with metadata (doc_name, page, score)
- `generated_answer`: LLM answer
- `model_answer`: Ground truth
- Full retrieval metrics (PageRec@5, ChunkRec@5, etc.)

## 🔧 Configuration Options

You can customize the script by editing [scripts/run_page_baseline.sh](../scripts/run_page_baseline.sh):

```bash
LLM_MODEL="qwen"           # or "llama"
EMBEDDING_MODEL="BAAI/bge-m3"
CHUNK_SIZE=1024            # Chunk size in characters
CHUNK_OVERLAP=128          # Overlap between chunks
TOP_K=5                    # Number of final chunks for generation
```

## 📈 Next Steps After Running

1. **Score the results** (if not auto-scored):
   ```bash
   python scripts/score_experiment.py \
     --input-file "outputs/page_baseline/YYYYMMDD/page_baseline_*.json"
   ```

2. **Compare with learned scorer**:
   ```bash
   # Run learned page scorer
   sbatch scripts/run_page_learned.sh
   
   # Compare results
   python scripts/compare_experiments.py \
     --baseline "outputs/page_baseline/.../page_baseline_*_scored.json" \
     --learned "outputs/page_learned/.../page_learned_*_scored.json"
   ```

3. **Run geometric analysis** (to understand retrieval difficulty):
   ```bash
   python scripts/geometric_analysis.py \
     --results-file "outputs/page_baseline/.../page_baseline_*_scored.json" \
     --embeddings-dir "analysis/geometric/page_baseline/embeddings" \
     --output-dir "analysis/geometric/page_baseline/analysis" \
     --method-name "Page-Then-Chunk Baseline"
   ```

## 🐛 Troubleshooting

**Issue**: "No pages found in vectorstore"
- **Fix**: Delete `vector_stores/chroma/pages_v12_text_baseline/` and rerun

**Issue**: "CUDA out of memory"  
- **Fix**: Reduce batch size or use smaller model

**Issue**: "PDF not found"
- **Fix**: Ensure PDFs are in `pdfs/` directory

## 🎯 Summary

Your baseline page-then-chunk pipeline:
- ✅ **Correctly implemented** in `src/experiments/page_retrieval.py`
- ✅ **Uses BGE-M3** for pages AND chunks (no learned scorer)
- ✅ **Retrieves 5 pages** → chunks them → ranks chunks → uses top-5 for generation
- ✅ **Ready to run** with `sbatch scripts/run_page_baseline.sh`
- ✅ **Compatible** with existing evaluation and analysis scripts

This provides your baseline to compare against the learned page scorer!
