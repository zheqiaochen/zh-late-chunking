<h1 align="center">
ZH Late Chunking
</h1>

<p align="center">
[English](README_en.md)
</p>

### 简介
这个项目实现了中文 [Late Chunking](https://jina.ai/news/late-chunking-in-long-context-embedding-models) 
的封装，用户传入一个包含多个待处理的文本段（text）的列表，类似：

```{python}
texts = [
    "谢良兴国县长冈乡塘石村人。生于一九一五年四月。一九三○年六月参加中国工农红军，同年七月加人中国共产主义青年团",
    "解放战争时期，任陕甘宁联防军政治部组织部副部长，联防军后勤部政治部主任，冀鲁豫军区政治部组织部部长",
    "因战伤截肢，人称“独脚将军。”获一级八一勋章、二级独立自由勋章、一级解放勋章和一级红星功勋荣誉章。"
]
```

使用LateEmbedding的main函数：

```{python}
embeddings = main(text)
```

最终返回经过late embedding的numpy数组：

```{python}
[array([-1.1230367 ,  1.0277416 , -0.6005747 ,  0.6233702 ,  0.05251494,
        -1.0940877 ,  0.1847056 , ...
```

默认使用`Alibaba-NLP/gte-multilingual-base`模型。
### 原理
1. 除了最后一条text以外，在每条 text 末尾加入特殊字符 “\</s>” 进行隔断，然后记录隔断所在位置
2. 把 text 合并在一起进行 embedding
3. 根据加入的隔断找回分割的 text，对每个分段内的 token embedding 求均值

### 感谢
[late-chunking-chinese项目](https://github.com/zhanshijinwat/late-chunking-chinese)
[late-chuning项目](https://github.com/jina-ai/late-chunking)