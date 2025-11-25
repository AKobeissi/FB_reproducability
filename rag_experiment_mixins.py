"""
Mixins that encapsulate the reusable pieces of the RAGExperiment class.

Breaking these helpers out keeps `rag_experiments.py` focused on the orchestration
logic while retaining the existing behavior.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from .rag_dependencies import (
        Document,
        RecursiveCharacterTextSplitter,
        build_chroma_store,
        create_faiss_store,
        retrieve_faiss_chunks,
    )
except ImportError:
    from rag_dependencies import (
        Document,
        RecursiveCharacterTextSplitter,
        build_chroma_store,
        create_faiss_store,
        retrieve_faiss_chunks,
    )


logger = logging.getLogger(__name__)


class ComponentTrackingMixin:
    """Shared helpers for recording and reporting component usage."""

    def register_component_usage(self, component: str, name: str, metadata: Optional[Dict[str, Any]] = None):
        self.component_usage[component] = {
            "name": name,
            "metadata": metadata or {}
        }

    def _describe_component(self, key: str, fallback: str) -> str:
        data = self.component_usage.get(key)
        if not data:
            return fallback
        if data.get("metadata"):
            details = ", ".join(f"{k}={v}" for k, v in data["metadata"].items())
            return f"{data['name']} ({details})"
        return data["name"]

    def _print_component_overview(self, stage: str = "current"):
        """Print human-readable summary of frameworks/components in use."""
        print("\n" + "=" * 80)
        print(f"Experiment Component Summary [{stage}]")
        print("=" * 80)
        print(f"Experiment Type : {self.experiment_type}")
        print(f"LLM Generator   : {self._describe_component('generator', f'pending ({self.llm_model_name})')}")
        print(f"Chunker         : {self._describe_component('chunker', 'pending')}")
        print(f"Embeddings      : {self._describe_component('embeddings', 'pending')}")
        vector_hint = "not required (closed/open book)" if self.experiment_type in {self.CLOSED_BOOK, self.OPEN_BOOK} else "pending"
        retriever_hint = "not required (closed/open book)" if self.experiment_type in {self.CLOSED_BOOK, self.OPEN_BOOK} else "pending"
        print(f"Vector Store    : {self._describe_component('vector_store', vector_hint)}")
        print(f"Retriever       : {self._describe_component('retriever', retriever_hint)}")
        print(f"Top-K           : {self.top_k}")
        print(f"Chunking Params : size={self.chunk_size}, overlap={self.chunk_overlap}")
        mode = "OpenAI API" if self.use_api else "Local HF weights"
        print(f"Generation Mode : {mode}")
        print("=" * 80 + "\n")

    def _set_progress_total(self, total: int):
        self.progress_total = total or 0
        self.progress_completed = 0
        if self.progress_total:
            self._print_progress(prefix="initialized")

    def notify_sample_complete(self, count: int = 1, note: Optional[str] = None):
        """Update and print progress for processed samples."""
        if count <= 0:
            return
        self.progress_completed += count
        self._print_progress(note=note)

    def _print_progress(self, note: Optional[str] = None, prefix: Optional[str] = None):
        total = self.progress_total or "?"
        prefix_text = f"{prefix} | " if prefix else ""
        msg = f"[Progress] {prefix_text}{self.progress_completed}/{total} samples processed"
        if note:
            msg += f" ({note})"
        print(msg)
        self.logger.info(msg)


class ChunkAndEvidenceMixin:
    """Helpers for chunking documents and normalizing gold evidence."""

    def _chunk_text_langchain(self, text, metadata: Dict[str, Any] = None) -> List[Document]:
        metadata = metadata or {}

        if isinstance(text, list):
            documents_input = []
            for doc in text:
                doc_meta = dict(metadata)
                doc_meta.update(doc.metadata or {})
                doc.metadata = doc_meta
                documents_input.append(doc)
            documents = self.text_splitter.split_documents(documents_input)
        else:
            if isinstance(text, (bytes, bytearray)):
                try:
                    text = text.decode('utf-8')
                except Exception:
                    text = text.decode('utf-8', errors='replace')

            if not text or len(text) == 0:
                return []

            documents = self.text_splitter.create_documents(
                texts=[str(text)],
                metadatas=[metadata]
            )

        chunk_lengths = [
            len(doc.page_content)
            if not isinstance(doc.page_content, (bytes, bytearray))
            else len(doc.page_content.decode('utf-8', errors='replace'))
            for doc in documents
        ]

        self.logger.info(f"\nChunking Statistics (LangChain):")
        self.logger.info(f"  Total chunks: {len(documents)}")
        self.logger.info(f"  Avg chunk size: {np.mean(chunk_lengths):.2f} chars")
        self.logger.info(f"  Min chunk size: {np.min(chunk_lengths)} chars")
        self.logger.info(f"  Max chunk size: {np.max(chunk_lengths)} chars")
        self.logger.info(f"  Median chunk size: {np.median(chunk_lengths):.2f} chars")

        self.logger.debug(f"\nFirst 3 chunks preview:")
        for i, doc in enumerate(documents[:3]):
            content_preview = doc.page_content
            if isinstance(content_preview, (bytes, bytearray)):
                content_preview = content_preview.decode('utf-8', errors='replace')
            self.logger.debug(f"  Chunk {i}: {content_preview[:100]}...")
            self.logger.debug(f"    Metadata: {doc.metadata}")

        return documents

    def _normalize_evidence(self, evidence: Any) -> List[str]:
        parts: List[str] = []
        if evidence is None:
            return parts

        if isinstance(evidence, (bytes, bytearray)):
            try:
                parts.append(evidence.decode('utf-8'))
            except Exception:
                parts.append(evidence.decode('utf-8', errors='replace'))
            return parts

        if isinstance(evidence, str):
            return [evidence]

        try:
            import numpy as _np
            if isinstance(evidence, _np.ndarray):
                for v in evidence.tolist():
                    if isinstance(v, (bytes, bytearray)):
                        try:
                            parts.append(v.decode('utf-8'))
                        except Exception:
                            parts.append(v.decode('utf-8', errors='replace'))
                    elif v is None:
                        continue
                    else:
                        parts.append(str(v))
                return [p for p in parts if p]
        except Exception:
            pass

        if isinstance(evidence, (list, tuple)):
            for v in evidence:
                if v is None:
                    continue
                if isinstance(v, (bytes, bytearray)):
                    try:
                        parts.append(v.decode('utf-8'))
                    except Exception:
                        parts.append(v.decode('utf-8', errors='replace'))
                else:
                    parts.append(str(v))
            return [p for p in parts if p]

        try:
            return [str(evidence)]
        except Exception:
            return []

    def _prepare_gold_evidence(self, evidence: Any) -> Tuple[List[Dict[str, Any]], str]:
        segments = self._coerce_evidence_segments(evidence)
        combined_text = "\n\n".join(
            seg.get("text", "")
            for seg in segments
            if seg.get("text")
        )
        return segments, combined_text

    def _coerce_evidence_segments(self, evidence: Any) -> List[Dict[str, Any]]:
        if evidence is None:
            return []

        raw = evidence
        if isinstance(evidence, (bytes, bytearray)):
            try:
                raw = evidence.decode("utf-8")
            except Exception:
                raw = evidence.decode("utf-8", errors="replace")

        if isinstance(raw, str):
            stripped = raw.strip()
            if stripped.startswith(("{", "[")):
                try:
                    raw = json.loads(stripped)
                except Exception:
                    try:
                        import ast
                        raw = ast.literal_eval(stripped)
                    except Exception:
                        raw = [stripped]
            else:
                return [{"text": stripped, "doc_name": None, "page": None, "page_text": None, "raw": stripped}]

        try:
            import numpy as _np
            if isinstance(raw, _np.ndarray):
                raw = raw.tolist()
        except Exception:
            pass

        if isinstance(raw, dict):
            entries = [raw]
        elif isinstance(raw, (list, tuple)):
            entries = list(raw)
        else:
            entries = [raw]

        segments: List[Dict[str, Any]] = []
        for entry in entries:
            if entry is None:
                continue
            if isinstance(entry, (bytes, bytearray)):
                try:
                    entry = entry.decode("utf-8")
                except Exception:
                    entry = entry.decode("utf-8", errors="replace")
            if isinstance(entry, str):
                segments.append({
                    "text": entry,
                    "doc_name": None,
                    "page": None,
                    "page_text": None,
                    "raw": entry,
                })
                continue

            if isinstance(entry, dict):
                text = (
                    entry.get("evidence_text")
                    or entry.get("text")
                    or entry.get("evidence_text_full_page")
                    or entry.get("excerpt")
                )
                if not text:
                    text = str(entry)
                segments.append({
                    "text": text,
                    "doc_name": entry.get("doc_name") or entry.get("document") or entry.get("doc"),
                    "page": entry.get("evidence_page_num") or entry.get("page") or entry.get("page_num"),
                    "page_text": entry.get("evidence_text_full_page"),
                    "raw": entry,
                })
                continue

            segments.append({
                "text": str(entry),
                "doc_name": None,
                "page": None,
                "page_text": None,
                "raw": entry,
            })

        return segments


class PromptMixin:
    """Prompt construction and answer generation helpers."""

    def _build_financebench_prompt(
        self,
        question: str,
        context: Optional[str] = None,
        mode: Optional[str] = None,
    ) -> str:
        question = (question or "").strip()
        context = (context or "").strip()
        prompt_mode = mode or self.experiment_type

        if context:
            context = context[: self.max_context_chars]

        if not context:
            return f"Answer this question: {question}"

        if prompt_mode in {self.OPEN_BOOK, "oracle"}:
            header = "Here is the relevant evidence that you need to answer the question:"
        else:
            header = "Here is the relevant filing that you need to answer the question:"

        return (
            f"Answer this question: {question}\n"
            f"{header}\n"
            "[START OF FILING]\n"
            f"{context}\n"
            "[END OF FILING]"
        )

    def _generate_answer(
        self,
        question: str,
        context: Optional[str] = None,
        mode: Optional[str] = None,
        return_prompt: bool = False,
    ) -> Any:
        prompt = self._build_financebench_prompt(question, context, mode=mode)

        if self.use_api:
            self._initialize_llm()

            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a helpful financial analyst assistant. "
                        "Answer strictly based on the question and the provided context "
                        "(if any) and avoid speculation."
                    ),
                },
                {"role": "user", "content": prompt},
            ]

            try:
                response = self.api_client.chat.completions.create(
                    model=self.llm_model_name,
                    messages=messages,
                    max_tokens=self.max_new_tokens,
                    temperature=0.0,
                )
                answer = (response.choices[0].message.content or "").strip()
            except Exception as e:
                self.logger.error(f"API generation failed: {e}")
                answer = ""
            return (answer, prompt) if return_prompt else answer

        self._initialize_llm()

        try:
            outputs = self.llm_pipeline(prompt)
        except TypeError:
            outputs = self.llm_pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        if not outputs:
            answer = ""
        else:
            text = outputs[0].get("generated_text", "")
            answer = (text or "").strip()

        return (answer, prompt) if return_prompt else answer


class VectorstoreMixin:
    """Helpers for delegating vector store creation/retrieval."""

    def _build_vectorstore_chroma(self, docs, embeddings=None):
        if build_chroma_store is None:
            raise RuntimeError("Chroma vectorstore helper not available; ensure vectorstore.py is present and importable.")
        try:
            return build_chroma_store(self, docs, embeddings=embeddings)
        except Exception as e:
            self.logger.error(f"_build_vectorstore_chroma delegated to vectorstore failed: {e}")
            raise

    def _create_vector_store_faiss(self, documents: List[Document], index_name: str = "default") -> Any:
        if create_faiss_store is None:
            raise RuntimeError("FAISS helper not available; ensure vectorstore.py is present and importable.")
        try:
            return create_faiss_store(self, documents, index_name=index_name)
        except Exception as e:
            self.logger.error(f"_create_vector_store_faiss delegated to vectorstore failed: {e}")
            raise

    def _retrieve_chunks_faiss(self, query: str, vector_store: Any, top_k: Optional[int] = None) -> List[Dict[str, Any]]:
        if retrieve_faiss_chunks is None:
            try:
                return retrieve_faiss_chunks(self, query, vector_store, top_k=top_k)
            except Exception:
                self.logger.warning("FAISS retrieval helper not available; no retrieval performed")
                return []
        try:
            return retrieve_faiss_chunks(self, query, vector_store, top_k=top_k)
        except Exception as e:
            self.logger.warning(f"FAISS retrieval helper failed: {e}")
            return []


class ResultsMixin:
    """Persistence and aggregation helpers."""

    def _create_skipped_results(
        self,
        samples: List[Dict[str, Any]],
        doc_name: str,
        doc_link: str,
        pdf_source: str,
        experiment_type: str,
        start_id: int,
        experiment=None,
    ) -> List[Dict[str, Any]]:
        results = []
        for i, sample in enumerate(samples):
            prompt = ""
            if experiment is not None:
                prompt = experiment._build_financebench_prompt(sample.get('question', ''), "", mode=experiment_type)
            results.append({
                'sample_id': start_id + i,
                'doc_name': doc_name,
                'doc_link': doc_link,
                'question': sample.get('question', ''),
                'reference_answer': sample.get('answer', ''),
                'gold_evidence': '',
                'gold_evidence_segments': [],
                'retrieved_chunks': [],
                'num_retrieved': 0,
                'context_length': 0,
                'generated_answer': '',
                'generation_length': 0,
                'experiment_type': experiment_type,
                'vector_store_type': None,
                'pdf_source': pdf_source,
                'final_prompt': prompt,
            })
        return results

    def _compute_aggregate_stats(self):
        self.logger.info("\n" + "=" * 80)
        self.logger.info("AGGREGATE STATISTICS")
        self.logger.info("=" * 80)

        if not self.results:
            self.logger.warning("No results to aggregate")
            return

        gen_lengths = [r['generation_length'] for r in self.results]
        self.logger.info(f"\nGenerated Answer Lengths:")
        self.logger.info(f"  Mean: {np.mean(gen_lengths):.2f} chars")
        self.logger.info(f"  Min: {np.min(gen_lengths)} chars")
        self.logger.info(f"  Max: {np.max(gen_lengths)} chars")
        self.logger.info(f"  Median: {np.median(gen_lengths):.2f} chars")

        retrieval_modes = [self.SINGLE_VECTOR, self.SHARED_VECTOR]
        if hasattr(self, "RANDOM_SINGLE"):
            retrieval_modes.append(self.RANDOM_SINGLE)

        if self.experiment_type in retrieval_modes:
            context_lengths = [r['context_length'] for r in self.results]
            self.logger.info(f"\nContext Lengths:")
            self.logger.info(f"  Mean: {np.mean(context_lengths):.2f} chars")
            self.logger.info(f"  Min: {np.min(context_lengths)} chars")
            self.logger.info(f"  Max: {np.max(context_lengths)} chars")

            retrieved_counts = [
                r.get('num_retrieved') for r in self.results if r.get('num_retrieved') is not None
            ]
            if retrieved_counts:
                self.logger.info(f"  Mean retrieved chunks: {np.mean(retrieved_counts):.2f}")
                self.logger.info(f"  Max retrieved chunks: {np.max(retrieved_counts)}")

        self.logger.info("\nNo automatic BLEU/ROUGE metrics computed during the run (evaluation is now post-hoc).")

        self.logger.info("=" * 80)

    def _save_results(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/{self.experiment_type}_{timestamp}.json"

        output_data = {
            'metadata': self.experiment_metadata,
            'num_samples': len(self.results),
            'framework': 'LangChain + Chroma',
            'results': self.results
        }

        def _to_json_serializable(obj):
            try:
                import numpy as _np
            except Exception:
                _np = None

            if obj is None:
                return None
            if isinstance(obj, (str, bool, int, float)):
                return obj
            if _np is not None and isinstance(obj, _np.ndarray):
                return obj.tolist()
            if _np is not None and isinstance(obj, _np.generic):
                try:
                    return obj.item()
                except Exception:
                    return str(obj)
            if isinstance(obj, dict):
                return {str(k): _to_json_serializable(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_to_json_serializable(v) for v in obj]
            if isinstance(obj, tuple):
                return [_to_json_serializable(v) for v in obj]
            if hasattr(obj, '__dict__'):
                try:
                    return _to_json_serializable(obj.__dict__)
                except Exception:
                    pass
            try:
                return str(obj)
            except Exception:
                return None

        serializable = _to_json_serializable(output_data)

        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        with open(filename, 'w') as f:
            json.dump(serializable, f, indent=2)

        self.logger.info(f"\nResults saved to: {filename}")
