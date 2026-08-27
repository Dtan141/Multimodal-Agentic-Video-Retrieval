from __future__ import annotations

from contextlib import nullcontext
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, AutoTokenizer


class Siglip2Embedder:
    """SigLIP2 wrapper used by both the offline embedding pipeline and query search."""

    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        device: str | None = None,
        use_fp16: bool = True,
        text_max_length: int = 64,
        load_image_processor: bool = True,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.use_fp16 = bool(use_fp16) and self.device.startswith("cuda")
        self.text_max_length = int(text_max_length)
        self.image_processor = (
            AutoImageProcessor.from_pretrained(model_id)
            if load_image_processor
            else None
        )

        self.tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        model_dtype = torch.float16 if self.use_fp16 else torch.float32
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
        if getattr(output, "pooler_output", None) is not None:
            return output.pooler_output
        if getattr(output, "image_embeds", None) is not None:
            return output.image_embeds
        if getattr(output, "text_embeds", None) is not None:
            return output.text_embeds
        if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
            return output[0]
        raise TypeError(f"Không lấy được pooled embedding từ output type: {type(output)}")

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features.float(), p=2, dim=-1)

    @torch.inference_mode()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        if not images:
            raise ValueError("encode_images() nhận list ảnh rỗng")
        if self.image_processor is None:
            raise RuntimeError("Image processor chưa được load cho instance này.")

        inputs = self.image_processor(
            images=[image.convert("RGB") for image in images],
            return_tensors="pt",
        )
        inputs = self._move_to_device(inputs)
        with self._autocast_context():
            output = self.model.get_image_features(**inputs)
        return self._normalize(self._extract_pooled_features(output))

    @torch.inference_mode()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("encode_texts() nhận list text rỗng")

        inputs = self.tokenizer(
            [str(text).strip().lower() for text in texts],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
        )
        inputs = self._move_to_device(inputs)
        with self._autocast_context():
            output = self.model.get_text_features(**inputs)
        return self._normalize(self._extract_pooled_features(output))
