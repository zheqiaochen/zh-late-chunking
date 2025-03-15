import re
import numpy as np
from typing import List
from transformers import AutoModel, AutoTokenizer

class LateEmbedding:
    def __init__(self, model_name: str="Alibaba-NLP/gte-multilingual-base", trust_remote_code: bool = True):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=trust_remote_code)

    def insert_split_token(self, input_text: List[str], split_token="</s>"):
        """
        在文本中插入 </s>
        """
        result = []
        for i, text in enumerate(input_text):
            # 如果已经有split_token则删除
            text = text.replace(split_token, '')
            
            # 除了最后一个元素外，其他元素末尾添加split_token
            if i < len(input_text) - 1:
                text = text + split_token     
            result.append(text)
        combined_text = "".join(result)
        return combined_text

    def chunk_by_sentences(self, input_text: str, split_token="</s>"):
        """
        1. 利用re查找 split_token 在文本中的位置，确定每个文本块的字符范围；
        2. 对整个文本进行 tokenize 并获得 offset mapping；
        3. 对于每个文本块，根据其字符范围在 offset mapping 中寻找对应的 token 范围。
        """
        # 用re查找 split_token 在文本中的位置，确定每个文本块的字符范围
        pattern = re.escape(split_token)
        matches = list(re.finditer(pattern, input_text))
        
        seg_char_spans = []
        prev = 0
        for m in matches:
            seg_char_spans.append((prev, m.start()))
            prev = m.end()
        # 最后一段
        if prev < len(input_text):
            seg_char_spans.append((prev, len(input_text)))
        
        segments = [input_text[start:end] for start, end in seg_char_spans]
        
        # 对整个文本进行 tokenize，并获得 offset mapping
        inputs = self.tokenizer(input_text, return_tensors='pt', return_offsets_mapping=True)
        token_offsets = inputs['offset_mapping'][0]
        
        # 对每个文本块，根据其字符范围找到对应的 token 索引范围
        seg_token_spans = []
        for seg_start, seg_end in seg_char_spans:
            token_start = None
            token_end = None
            for i, (tok_start, tok_end) in enumerate(token_offsets):
                # 找到第一个结束位置大于段落起始位置的 token 作为 token_start
                if token_start is None and tok_end > seg_start:
                    token_start = i
                # token_end 更新为最后一个起始位置在段落结束之前的 token
                if tok_start < seg_end:
                    token_end = i
                if tok_start >= seg_end:
                    break
            # 若找不到则为 0
            if token_start is None:
                token_start = 0
            if token_end is None:
                token_end = 0
            seg_token_spans.append((token_start, token_end))
        
        return segments, seg_token_spans

    def chunked_pooling(self, model_output, span_annotation: list, max_length=None):
        """
        对token embedding序列做mean pooling
        """
        token_embeddings = model_output[0]
        outputs = []
        for embeddings, annotations in zip(token_embeddings, span_annotation):
            if (
                max_length is not None
            ):  
                annotations = [
                    (start, min(end, max_length - 1))
                    for (start, end) in annotations
                    if start < (max_length - 1)
                ]
            # 对每个span范围内的embeddings进行平均池化
            pooled_embeddings = [
                embeddings[start:end].sum(dim=0) / (end - start)
                for start, end in annotations
                if (end - start) >= 1
            ]
            # 将池化后的embeddings转换为numpy数组
            pooled_embeddings = [
                embedding.detach().cpu().numpy() for embedding in pooled_embeddings
            ]
            outputs.append(pooled_embeddings)

        return outputs

    def main(self, texts: List[str]):
        processed_text = self.insert_split_token(texts)
        chunks, span_annotations = self.chunk_by_sentences(processed_text)
        inputs = self.tokenizer(processed_text, return_tensors='pt')
        model_output = self.model(**inputs)
        embeddings = self.chunked_pooling(model_output, [span_annotations])[0]

        return embeddings

# if __name__ == "__main__":
#     sample_texts = ["第1个测试文本", "第2个测试文本", "第3个测试文本"]
#     embedder = LateEmbedding()
#     embeddings = embedder.main(sample_texts)
#     for i, emb in enumerate(embeddings):
#         print(f"文本 {i} 的 embedding 维度: {emb.shape}")
