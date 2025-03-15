<h1 align="center">
ZH Late Chunking
</h1>

<p align="center">
[English](README_en.md)
</p>

### Introduction
This project implements a wrapper for Chinese [Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models). Users provide a list of text segments to be processed, for example:

    texts = [
        "谢良兴国县长冈乡塘石村人。生于一九一五年四月。一九三○年六月参加中国工农红军，同年七月加人中国共产主义青年团",
        "解放战争时期，任陕甘宁联防军政治部组织部副部长，联防军后勤部政治部主任，冀鲁豫军区政治部组织部部长",
        "因战伤截肢，人称“独脚将军。”获一级八一勋章、二级独立自由勋章、一级解放勋章和一级红星功勋荣誉章。"
    ]

Use the `main` function of `LateEmbedding` as follows:

    embeddings = main(text)

This function ultimately returns a numpy array produced by the late embedding process:

    [array([-1.1230367 ,  1.0277416 , -0.6005747 ,  0.6233702 ,  0.05251494,
            -1.0940877 ,  0.1847056 , ...]
            
By default, the model used is `Alibaba-NLP/gte-multilingual-base`.

### How
1. For every text except the last one, a special token `</s>` is appended at the end to serve as a delimiter, and the positions of these delimiters are recorded.
2. The texts are then concatenated together for embedding.
3. The inserted delimiters are used to recover the original text segmentation, and the token embeddings within each segment are averaged.

### Acknowledgements
- [late-chunking-chinese project](https://github.com/zhanshijinwat/late-chunking-chinese)
- [late-chunking project](https://github.com/jina-ai/late-chunking)
