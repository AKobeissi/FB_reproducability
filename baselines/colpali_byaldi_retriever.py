"""
Visual Page Retriever — CLIP ViT-L/14 + FAISS
==============================================
Replaces the byaldi/colpali-engine stack (irreconcilable transformers version
conflicts) with a dependency-conflict-free approach using packages already
present in the venv.

Pipeline
--------
1. Scan all PDFs in pdf_dir and build a manifest of (doc_name, page_num, pdf_path)
2. Render every page to a PIL image via PyMuPDF
3. Encode each page with CLIP ViT-L/14 (openai/clip-vit-large-patch14, 768-dim)
4. L2-normalise embeddings → IndexFlatIP  ≡  cosine similarity search
5. At query time: encode the text query with the CLIP text tower → top-K search
6. Re-render matched pages on demand for the VLM generator

Cached artefacts (index_root/index_name/)
-----------------------------------------
  clip_faiss.index      FAISS flat-inner-product index
  page_manifest.json    [{doc_name, page_num, pdf_path}, …] in index order
"""

from __future__ import annotations

import base64
import json
import logging
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)

CLIP_MODEL  = "openai/clip-vit-large-patch14"
RENDER_DPI  = 150
BATCH_SIZE  = 32


class CLIPVisualRetriever:
    """
    CLIP-based visual page retriever.

    Drop-in replacement for the original ColPaliByaldiRetriever —
    same build_index / search / free interface and same return schema.
    """

    def __init__(
        self,
        pdf_dir: str,
        index_name: str = "financebench_clip_vl",
        index_root: str = ".clip_index",
        clip_model: str = CLIP_MODEL,
        device: str = "cuda",
    ) -> None:
        self.pdf_dir    = Path(pdf_dir)
        self.index_dir  = Path(index_root) / index_name
        self.clip_model = clip_model
        self.device     = device

        self._clip      = None   # CLIPModel
        self._proc      = None   # CLIPProcessor
        self._index     = None   # faiss.Index
        self._manifest: List[Dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_index(self, overwrite: bool = False) -> None:
        """Build (or load from cache) the CLIP FAISS index."""
        idx_path = self.index_dir / "clip_faiss.index"
        man_path = self.index_dir / "page_manifest.json"

        if idx_path.exists() and man_path.exists() and not overwrite:
            logger.info("Loading cached CLIP index from %s …", self.index_dir)
            self._load_from_disk(idx_path, man_path)
            return

        self.index_dir.mkdir(parents=True, exist_ok=True)
        self._load_clip()

        # --- Step 1: build manifest by scanning all PDFs ---
        pdf_files = sorted(self.pdf_dir.glob("*.pdf"))
        logger.info("Scanning %d PDFs to build page manifest …", len(pdf_files))
        manifest: List[Dict] = []
        for pdf_path in pdf_files:
            try:
                import pymupdf as fitz
                doc = fitz.open(str(pdf_path))
                n_pages = doc.page_count
                doc.close()
            except Exception as exc:
                logger.warning("Cannot open %s: %s", pdf_path.name, exc)
                continue
            for p in range(1, n_pages + 1):   # 1-indexed
                manifest.append({
                    "doc_name": pdf_path.stem,
                    "page_num": p,
                    "pdf_path": str(pdf_path),
                })
        logger.info("Manifest: %d pages across %d PDFs.", len(manifest), len(pdf_files))
        if not manifest:
            raise RuntimeError(
                "No pages could be read from any PDF. "
                "Check that PDFs exist in pdf_dir and that pymupdf can open them."
            )

        # --- Step 2: encode in batches ---
        all_embs: List[np.ndarray] = []
        n_batches = (len(manifest) + BATCH_SIZE - 1) // BATCH_SIZE
        for b in range(n_batches):
            batch = manifest[b * BATCH_SIZE : (b + 1) * BATCH_SIZE]
            pil_imgs = []
            for m in batch:
                try:
                    pil_imgs.append(
                        self._render_page(Path(m["pdf_path"]), m["page_num"])
                    )
                except Exception as exc:
                    logger.debug("Render fail %s p%d: %s", m["doc_name"], m["page_num"], exc)
                    from PIL import Image
                    pil_imgs.append(Image.new("RGB", (224, 224), (200, 200, 200)))

            batch_emb = self._encode_images(pil_imgs)   # (B, D)
            all_embs.append(batch_emb)

            if b % 20 == 0:
                done = min((b + 1) * BATCH_SIZE, len(manifest))
                logger.info("  Encoded %d / %d pages …", done, len(manifest))

        emb_matrix = np.vstack(all_embs).astype(np.float32)

        # --- Step 3: L2-normalise then build FAISS IndexFlatIP ---
        norms = np.linalg.norm(emb_matrix, axis=1, keepdims=True)
        emb_matrix /= np.maximum(norms, 1e-8)

        import faiss
        dim   = emb_matrix.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(emb_matrix)
        logger.info("FAISS index built: %d vectors, dim=%d", index.ntotal, dim)

        # --- Persist ---
        faiss.write_index(index, str(idx_path))
        with open(man_path, "w") as fh:
            json.dump(manifest, fh)
        logger.info("Saved index + manifest to %s", self.index_dir)

        self._index    = index
        self._manifest = manifest

    def load_index(self) -> None:
        idx_path = self.index_dir / "clip_faiss.index"
        man_path = self.index_dir / "page_manifest.json"
        self._load_from_disk(idx_path, man_path)

    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for the top_k pages most relevant to *query*.

        Returns
        -------
        List of dicts:
            doc_name     : str
            page         : int   (1-indexed, matches evidence_page_num in FinanceBench)
            score        : float (cosine similarity)
            base64_image : str | None  (raw JPEG base64, no data-URI prefix)
        """
        if self._index is None:
            raise RuntimeError("Index not loaded. Call build_index() first.")
        if self._clip is None:
            self._load_clip()

        q_emb = self._encode_text(query)   # (1, D)

        import faiss
        scores, idxs = self._index.search(
            q_emb, min(top_k, self._index.ntotal)
        )

        results = []
        for score, i in zip(scores[0], idxs[0]):
            if i < 0:
                continue
            m  = self._manifest[i]
            b64 = self._page_b64(Path(m["pdf_path"]), m["page_num"])
            results.append({
                "doc_name":    m["doc_name"],
                "page":        m["page_num"],
                "score":       float(score),
                "base64_image": b64,
            })
        return results

    def free(self) -> None:
        self._clip  = None
        self._proc  = None
        self._index = None
        self._manifest = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("CLIPVisualRetriever freed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_clip(self) -> None:
        from transformers import CLIPModel, CLIPProcessor
        logger.info("Loading CLIP '%s' …", self.clip_model)
        self._proc = CLIPProcessor.from_pretrained(self.clip_model)
        self._clip = CLIPModel.from_pretrained(
            self.clip_model, torch_dtype=torch.float16
        ).to(self.device)
        self._clip.eval()
        logger.info("CLIP ready on %s.", self.device)

    def _load_from_disk(self, idx_path: Path, man_path: Path) -> None:
        import faiss
        self._index = faiss.read_index(str(idx_path))
        with open(man_path) as fh:
            self._manifest = json.load(fh)
        logger.info(
            "CLIP index loaded: %d pages from %s",
            self._index.ntotal, self.index_dir,
        )

    def _render_page(self, pdf_path: Path, page_num: int):
        """Render page_num (1-indexed) → RGB PIL Image."""
        import pymupdf as fitz
        from PIL import Image
        doc = fitz.open(str(pdf_path))
        mat = fitz.Matrix(RENDER_DPI / 72, RENDER_DPI / 72)
        pix = doc[page_num - 1].get_pixmap(matrix=mat, colorspace=fitz.csRGB)
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        doc.close()
        return img

    def _encode_images(self, pil_images) -> np.ndarray:
        """(N, D) float32 image embeddings."""
        inputs = self._proc(
            images=pil_images, return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            feats = self._clip.get_image_features(**inputs)
        return feats.cpu().float().numpy()

    def _encode_text(self, text: str) -> np.ndarray:
        """(1, D) float32, L2-normalised text embedding."""
        inputs = self._proc(
            text=[text], return_tensors="pt", padding=True, truncation=True
        ).to(self.device)
        with torch.no_grad():
            feats = self._clip.get_text_features(**inputs)
        emb = feats.cpu().float().numpy()
        emb /= np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8)
        return emb.astype(np.float32)

    def _page_b64(self, pdf_path: Path, page_num: int) -> Optional[str]:
        try:
            img = self._render_page(pdf_path, page_num)
            buf = BytesIO()
            img.save(buf, format="JPEG", quality=85)
            return base64.b64encode(buf.getvalue()).decode("ascii")
        except Exception as exc:
            logger.debug("b64 fail %s p%d: %s", pdf_path.name, page_num, exc)
            return None
