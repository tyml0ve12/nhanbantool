"""So nghia cau giua 2 ngon ngu khac nhau bang model nhung cau da ngon ngu
(paraphrase-multilingual-MiniLM-L12-v2, ban ONNX nen ~120 MB, 50+ ngon ngu).

Chay bang onnxruntime + tokenizers (da co san vi faster-whisper phu thuoc) -
khong can torch. Model nam trong models/ canh app (dong goi san khi phat hanh).
"""
import os

import numpy as np

from core.speech import get_models_dir

REPO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
ONNX_FILE = "onnx/model_quint8_avx2.onnx"
FILES = (ONNX_FILE, "tokenizer.json")
MAX_TOKENS = 128


def model_dir() -> str:
    return os.path.join(get_models_dir(), "multilingual-minilm")


def is_downloaded() -> bool:
    return all(os.path.isfile(os.path.join(model_dir(), f)) for f in FILES)


def download():
    from huggingface_hub import hf_hub_download
    for f in FILES:
        hf_hub_download(REPO, f, local_dir=model_dir())


class Embedder:
    def __init__(self):
        import onnxruntime as ort
        from tokenizers import Tokenizer

        if not is_downloaded():
            download()
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(os.path.join(model_dir(), ONNX_FILE), opts,
                                            providers=["CPUExecutionProvider"])
        self.input_names = {i.name for i in self.session.get_inputs()}
        self.tokenizer = Tokenizer.from_file(os.path.join(model_dir(), "tokenizer.json"))
        self.tokenizer.enable_truncation(MAX_TOKENS)
        self.tokenizer.enable_padding()

    def encode_token_sums(self, texts, batch_size: int = 128):
        """Tra ve (tong vector token (n, dim), so token (n,)) cua tung text -
        cho phep ghep nhieu doan ngan thanh vector cua ca doan dai ma khong
        phai chay lai model (mean pooling = tong / so token).
        Xep text theo do dai truoc khi chia lo -> it token dem thua, nhanh ~2 lan."""
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
        sums = np.zeros((len(texts), 384), dtype=np.float32)
        counts = np.zeros(len(texts), dtype=np.float32)
        for i in range(0, len(order), batch_size):
            idx = order[i:i + batch_size]
            batch = [texts[k] if texts[k].strip() else "." for k in idx]
            enc = self.tokenizer.encode_batch(batch)
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.input_names:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]
            m = mask[..., None].astype(np.float32)
            sums[idx] = (hidden * m).sum(1)
            counts[idx] = m.sum((1, 2))
        return sums, counts

    def encode(self, texts, batch_size: int = 128) -> np.ndarray:
        """Tra ve ma tran (len(texts), dim) da chuan hoa (dot = cosine)."""
        sums, counts = self.encode_token_sums(texts, batch_size)
        vec = sums / np.maximum(counts, 1e-9)[:, None]
        vec /= np.maximum(np.linalg.norm(vec, axis=1, keepdims=True), 1e-9)
        return vec
