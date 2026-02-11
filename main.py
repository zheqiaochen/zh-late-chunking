import re
from typing import List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


class LateEmbedding:
    def __init__(
        self,
        model_name: str = "Alibaba-NLP/gte-multilingual-base",
        trust_remote_code: bool = True,
        split_token: str = "<<<SPLIT>>>",
        device: Optional[str] = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)

        if not getattr(self.tokenizer, "is_fast", False):
            raise ValueError("需要 fast tokenizer 才能使用 return_offsets_mapping。")

        # split_token 不应该是 special token
        specials = set(self.tokenizer.all_special_tokens or [])
        if split_token in specials:
            raise ValueError(f"split_token={split_token} 是 tokenizer 的 special token，请换一个不冲突的分隔符。")

        self.split_token = split_token

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)
        self.model.eval()

    def insert_split_token(self, texts: List[str]) -> str:
        """把 texts 拼成一个字符串，中间插入 split_token，便于后续按字符位置切段。"""
        cleaned = [t.replace(self.split_token, "") for t in texts]
        return self.split_token.join(cleaned)

    def _find_char_spans(self, input_text: str) -> List[Tuple[int, int]]:
        """根据 split_token 的字符位置，得到每段的 (char_start, char_end)（不含 split_token 本身）。"""
        pattern = re.escape(self.split_token)
        matches = list(re.finditer(pattern, input_text))

        spans = []
        prev = 0
        for m in matches:
            spans.append((prev, m.start()))
            prev = m.end()
        spans.append((prev, len(input_text)))
        # 去掉空段（比如连续 split_token）
        return [(s, e) for s, e in spans if e > s]

    @staticmethod
    def _is_special_offset(off: Tuple[int, int]) -> bool:
        return off[0] == 0 and off[1] == 0

    def _char_span_to_token_span(
        self,
        char_span: Tuple[int, int],
        offsets: List[Tuple[int, int]],
        max_len: int,
    ) -> Optional[Tuple[int, int]]:
        """把字符区间映射到 token 区间 [start, end)；找不到则返回 None。"""
        seg_start, seg_end = char_span
        if seg_end <= seg_start:
            return None

        token_start = None
        token_end_excl = None

        for i, (tok_start, tok_end) in enumerate(offsets):
            if i >= max_len:
                break
            if self._is_special_offset((tok_start, tok_end)):
                continue

            if token_start is None and tok_end > seg_start:
                token_start = i
            if tok_start < seg_end:
                token_end_excl = i + 1 
            if tok_start >= seg_end:
                break

        if token_start is None or token_end_excl is None or token_end_excl <= token_start:
            return None
        return (token_start, min(token_end_excl, max_len))

    def main(self, texts: List[str], max_length: Optional[int] = None) -> List[np.ndarray]:
        processed_text = self.insert_split_token(texts)
        char_spans = self._find_char_spans(processed_text)

        # 一次 tokenize：offsets 和真正进模型的 input 完全一致
        enc = self.tokenizer(
            processed_text,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=max_length,
        )
        offsets = enc.pop("offset_mapping")[0].tolist()  # [(start,end), ...]
        enc = {k: v.to(self.device) for k, v in enc.items()}

        with torch.no_grad():
            out = self.model(**enc)
            token_emb = out.last_hidden_state[0]  # [seq_len, hidden]

        seq_len = token_emb.shape[0]
        token_spans = []
        for cs in char_spans:
            ts = self._char_span_to_token_span(cs, offsets, seq_len)
            if ts is not None:
                token_spans.append(ts)

        # mean pooling
        pooled = []
        for start, end in token_spans:
            vec = token_emb[start:end].mean(dim=0)
            pooled.append(vec.detach().cpu().numpy())

        return pooled


# 示例
if __name__ == "__main__":
    sample_texts = ["第1个测试文本", "第2个测试文本", "第3个测试文本"]
    embedder = LateEmbedding()
    embs = embedder.main(sample_texts, max_length=512)
    for i, e in enumerate(embs):
        print(i, e.shape)
