"""
Qwen2-VL-7B-Instruct visual generator for the ColPali-Byaldi Visual RAG pipeline.

Takes a text query + a list of base64-encoded page images and produces a grounded
financial answer using only the visual evidence.

Design
------
- Model  : Qwen/Qwen2-VL-7B-Instruct
- Dtype  : torch.bfloat16  (fast + memory-efficient on Ampere+ GPUs)
- Mapping: device_map="auto"  (handles multi-GPU or single-GPU automatically)
- Images : passed as data-URI strings so the processor handles decoding natively
           via qwen_vl_utils.process_vision_info (falling back to PIL if the util
           is unavailable)

Usage
-----
    gen = QwenVLGenerator()
    gen.load()
    answer = gen.generate_answer(query="What is net revenue?", base64_images=[b64_str, …])
    gen.free()
"""

from __future__ import annotations

import base64
import logging
from io import BytesIO
from typing import List, Optional

import torch

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a senior financial analyst. "
    "You will be shown one or more pages from a corporate financial filing. "
    "Answer the user's question using ONLY the information visible in the provided document images. "
    "Be concise and precise. "
    "If the answer requires a number, state it exactly as it appears in the document, including units. "
    "If the answer cannot be determined from the images, respond with: "
    "'I cannot determine the answer from the provided document pages.'"
)


class QwenVLGenerator:
    """Wrapper around Qwen2-VL-7B-Instruct for visual document QA."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-7B-Instruct",
        max_new_tokens: int = 512,
        min_pixels: int = 256 * 28 * 28,
        max_pixels: int = 1280 * 28 * 28,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

        self._model = None
        self._processor = None
        self._use_qwen_utils = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load model and processor onto GPU (bfloat16, device_map=auto)."""
        from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

        logger.info("Loading Qwen2-VL processor from '%s' …", self.model_name)
        try:
            # Try with pixel constraints (supported in transformers >= 4.45)
            self._processor = AutoProcessor.from_pretrained(
                self.model_name,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels,
            )
        except TypeError:
            self._processor = AutoProcessor.from_pretrained(self.model_name)

        logger.info("Loading Qwen2-VL model (bfloat16, device_map=auto) …")
        self._model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self._model.eval()

        # Check if qwen_vl_utils is available for process_vision_info
        try:
            from qwen_vl_utils import process_vision_info as _pvi  # noqa: F401

            self._use_qwen_utils = True
            logger.info("qwen_vl_utils found — using process_vision_info.")
        except ImportError:
            self._use_qwen_utils = False
            logger.info(
                "qwen_vl_utils not found — using PIL fallback for image loading."
            )

        logger.info("Qwen2-VL ready.")

    def free(self) -> None:
        """Delete model/processor and release GPU memory."""
        self._model = None
        self._processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Qwen2-VL generator freed.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def generate_answer(
        self,
        query: str,
        base64_images: List[Optional[str]],
    ) -> str:
        """Generate an answer grounded in the provided page images.

        Parameters
        ----------
        query          : The user's financial question.
        base64_images  : List of raw base64 strings (no data-URI prefix).
                         None entries are skipped gracefully.

        Returns
        -------
        The generated answer as a stripped string.
        """
        if self._model is None or self._processor is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        valid_images = [b for b in base64_images if b]
        if not valid_images:
            logger.warning("No images provided — returning fallback answer.")
            return "I cannot determine the answer from the provided document pages."

        # Build conversation content: images first, then question
        content = []
        for b64 in valid_images:
            content.append(
                {"type": "image", "image": f"data:image/jpeg;base64,{b64}"}
            )
        content.append(
            {
                "type": "text",
                "text": f"Question: {query}",
            }
        )

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": content},
        ]

        # Apply chat template to get the text prompt
        text_prompt = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Process vision info (images)
        if self._use_qwen_utils:
            from qwen_vl_utils import process_vision_info

            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self._processor(
                text=[text_prompt],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
        else:
            # PIL fallback — decode base64 → PIL Images
            pil_images = self._b64_list_to_pil(valid_images)
            inputs = self._processor(
                text=[text_prompt],
                images=pil_images if pil_images else None,
                padding=True,
                return_tensors="pt",
            )

        # Move to the same device as the model
        device = next(self._model.parameters()).device
        inputs = inputs.to(device)

        with torch.no_grad():
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )

        # Trim the prompt tokens from the output
        generated_ids_trimmed = [
            out_ids[len(in_ids):]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_texts = self._processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return output_texts[0].strip()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _b64_list_to_pil(b64_list: List[str]):
        """Decode a list of raw base64 JPEG strings to PIL Images."""
        from PIL import Image

        images = []
        for b64 in b64_list:
            try:
                img_bytes = base64.b64decode(b64)
                img = Image.open(BytesIO(img_bytes)).convert("RGB")
                images.append(img)
            except Exception as exc:
                logger.debug("Failed to decode base64 image: %s", exc)
        return images
