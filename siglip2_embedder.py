from __future__ import annotations

from typing import Any, Iterable

import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoModel, AutoImageProcessor


class Siglip2Embedder:
    def __init__(
        self,
        model_id: str = "google/siglip2-base-patch16-224",
        device: str | None = None,
        batch_size: int = 8,
        use_fp16: bool = True,
        text_max_length: int = 64,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.use_fp16 = use_fp16 and self.device.startswith("cuda")
        self.text_max_length = text_max_length

        dtype = torch.float16 if self.use_fp16 else torch.float32
        print(f"🚀 Loading {model_id} on {self.device} | fp16={self.use_fp16}")

        self.processor = AutoImageProcessor.from_pretrained(model_id)

        self.model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if self.use_fp16 else torch.float32,
        ).to(self.device)

        self.model.eval()

    def _move_to_device(self, inputs: Any) -> Any:
        """BatchFeature/BatchEncoding thường có .to(device), nhưng fallback dict cho chắc."""
        if hasattr(inputs, "to"):
            return inputs.to(self.device)

        if isinstance(inputs, dict):
            return {
                k: v.to(self.device) if hasattr(v, "to") else v
                for k, v in inputs.items()
            }

        return inputs

    @staticmethod
    def _extract_pooled_features(output: Any) -> torch.Tensor:
        """
        Một số version Transformers trả thẳng Tensor,
        một số version trả BaseModelOutputWithPooling.
        Retrieval cần pooled embedding, ưu tiên pooler_output.
        """
        if isinstance(output, torch.Tensor):
            return output

        if hasattr(output, "pooler_output") and output.pooler_output is not None:
            return output.pooler_output

        if hasattr(output, "image_embeds") and output.image_embeds is not None:
            return output.image_embeds

        if hasattr(output, "text_embeds") and output.text_embeds is not None:
            return output.text_embeds

        if isinstance(output, tuple) and len(output) > 0:
            first = output[0]
            if isinstance(first, torch.Tensor):
                return first

        raise TypeError(f"Không lấy được pooled embedding tensor từ output type: {type(output)}")

    @staticmethod
    def _normalize(features: torch.Tensor) -> torch.Tensor:
        return F.normalize(features.float(), p=2, dim=-1)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> torch.Tensor:
        """
        Embed danh sách PIL images.

        Returns:
            torch.Tensor shape [batch_size, dim], dtype float32, L2-normalized.
        """
        if not images:
            raise ValueError("encode_images() nhận list ảnh rỗng")

        rgb_images = [img.convert("RGB") for img in images]

        inputs = self.processor(
            images=rgb_images,
            return_tensors="pt",
        )
        inputs = self._move_to_device(inputs)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.use_fp16,
        ):
            output = self.model.get_image_features(**inputs)

        features = self._extract_pooled_features(output)
        return self._normalize(features)

    @torch.no_grad()
    def encode_image_paths(self, image_paths: list[str]) -> torch.Tensor:
        """
        Convenience method cho query image nhỏ/lẻ.
        Với pipeline batch lớn, nên tự load ảnh trong pipeline để xử lý lỗi file hỏng tốt hơn.
        """
        images: list[Image.Image] = []
        for path in image_paths:
            with Image.open(path) as img:
                images.append(img.convert("RGB").copy())

        return self.encode_images(images)

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        """
        Embed danh sách query text.

        Returns:
            torch.Tensor shape [batch_size, dim], dtype float32, L2-normalized.
        """
        if not texts:
            raise ValueError("encode_texts() nhận list text rỗng")

        clean_texts = [str(t).strip() for t in texts]

        inputs = self.processor(
            text=clean_texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.text_max_length,
        )
        inputs = self._move_to_device(inputs)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=self.use_fp16,
        ):
            output = self.model.get_text_features(**inputs)

        features = self._extract_pooled_features(output)
        return self._normalize(features)