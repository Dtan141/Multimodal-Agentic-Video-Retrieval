"""SigLIP2 text/image embedder.

Moved verbatim (with logging instead of ``print``) from the repo-root
``siglip2_embedder.py`` so the whole codebase imports one implementation. This
also fixes the broken ``from siglip2_embedder_v1 import Siglip2Embedder`` import
in ``test/rerank.py`` / ``vector_embeddings/embed_vector.py``.

Text and image features are L2-normalized, so cosine similarity == dot product.
"""

from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer

from aic_retrieval.logging_conf import get_logger

logger = get_logger(__name__)


class Siglip2Embedder:
    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        device: str | None = None,
        use_fp16: bool = True,
        text_max_length: int = 64,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = bool(use_fp16) and self.device.startswith("cuda")
        self.text_max_length = text_max_length

        model_dtype = torch.float16 if self.use_fp16 else torch.float32

        logger.info("Loading %s on %s | fp16=%s", model_id, self.device, self.use_fp16)

        self.image_processor = AutoImageProcessor.from_pretrained(model_id)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        self.model = AutoModel.from_pretrained(model_id, torch_dtype=model_dtype).to(self.device)
        self.model.eval()

    def _move_to_device(self, inputs: Any) -> Any:
        if hasattr(inputs, "to"):
            return inputs.to(self.device)
        if isinstance(inputs, dict):
            return {
                key: value.to(self.device) if hasattr(value, "to") else value
                for key, value in inputs.items()
            }
        return inputs

    def _autocast_context(self):
        if not self.use_fp16:
            return nullcontext()
        return torch.autocast(device_type="cuda", dtype=torch.float16)

    @staticmethod
    def _extract_pooled_features(output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output
        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds
        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError(f"Could not extract pooled embedding tensor from output type: {type(output)}")

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features.float(), p=2, dim=-1)

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("encode_images() received an empty image list")

        rgb_images = [image.convert("RGB") for image in images]
        inputs = self.image_processor(images=rgb_images, return_tensors="pt")
        inputs = self._move_to_device(inputs)

        with self._autocast_context():
            output = self.model.get_image_features(**inputs)

        return self._normalize(self._extract_pooled_features(output))

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("encode_texts() received an empty text list")

        clean_texts = [str(text).strip().lower() for text in texts]
        inputs = self.tokenizer(
            clean_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
        )
        inputs = self._move_to_device(inputs)

        with self._autocast_context():
            output = self.model.get_text_features(**inputs)

        return self._normalize(self._extract_pooled_features(output))
