# Gensim FastText 实现导览

本文是针对项目所用 Gensim 4.3.3 的开发者阅读指南，说明本项目调用了哪些组件，以及应从哪里查看它们的实现。运行 `python -m src.source_map` 可打印当前安装版本的源码行号。Cython 源码 `gensim/models/fasttext_inner.pyx` 可在 Conda 环境或 [Gensim 上游仓库](https://github.com/piskvorky/gensim/blob/4.3.3/gensim/models/fasttext_inner.pyx)中查看。

## 两项相关的 FastText 任务

| 参考资料 | 目标 | 本项目的对应部分 |
| --- | --- | --- |
| [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) | 利用字符 n-gram 和上下文学习词表示 | 使用 `sg=1` 的 `gensim.models.FastText` 训练词向量，再将向量用于分类 |
| [Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/) | 训练监督式文本分类器，包含词 n-gram 特征和标签预测层 | 只采用聚合文档向量的思路；标签分类器由 scikit-learn 单独训练 |

在 Gensim 4.3.3 中，`FastText.__init__` 不支持 `word_ngrams != 1`。`min_n` 和 `max_n` 指的是**词内部的字符** n-gram 长度，而分类论文中的词 n-gram 指的是**相邻词组成的序列**。将 Gensim 的 `hs` 设为 `1`，改变的是*词向量训练目标*，不会加入分类论文中的标签层次 softmax。两者名称相近，但作用于不同任务。

## 源码调用路径

1. `FastText.__init__`（`fasttext.py`）检查 `word_ngrams`，创建 `FastTextKeyedVectors`，并将通用参数传给 `Word2Vec.__init__`。
2. `Word2Vec.build_vocab` 扫描词频，并按 `min_count` 过滤。FastText 的 `FastTextKeyedVectors.recalc_char_ngram_buckets` 为词表中的每个词建立经过哈希的子词索引。`FastText.estimate_memory` 估算词表及 `bucket × vector_size` 子词向量所需的内存。
3. `Word2Vec.train` 安排训练批次、轮次和学习率衰减。`FastText._do_train_job` 调用 `fasttext_inner.pyx` 中编译后的 `train_batch_any`。当 `sg=1, hs=0, negative=5` 时，skip-gram 负采样路径会更新词向量、子词向量和输出向量。
4. `ft_ngram_hashes`（`fasttext.py`）调用 `compute_ngrams_bytes` 和 `ft_hash_bytes`（`fasttext_inner.pyx`），再对哈希值取模 `bucket`。Cython 函数为词加入 `<`、`>` 边界符，并沿 UTF-8 字符的起始位置遍历。对中文尤其要注意：n-gram 的长度按字符计算，而不是按原始 UTF-8 字节计算。
5. `FastTextKeyedVectors.adjust_vectors` 在训练后合成词表内词的词向量与子词向量。`__contains__` 检查是否可以返回向量。`get_vector` 对词表内词返回已合成的向量，对未登录词（OOV）则用哈希子词构造向量。
6. `_save_specials` 和 `_load_specials` 在模型保存、加载时处理派生向量缓存、哈希兼容性及缓存重建。项目保存完整的 `.model`，以便之后继续训练或检查向量。

[子词向量论文](https://arxiv.org/abs/1607.04606)用字符 n-gram 表示词，并通过上下文预测学习表示。Gensim 优化后的 Cython 代码还包含具体的缩放和缓存处理；对照论文时，应沿训练和推断路径检查，不能把论文公式当作逐行伪代码。`FastTextKeyedVectors.get_vector` 能在存在子词哈希时返回 OOV 向量，但这**不能证明向量具有良好的语义质量**。桶太小还可能使无关子词发生大量哈希冲突。

## 参数与实现的对应关系

| 参数 | 在源码中的作用 | 本项目的设置与检查方式 |
| --- | --- | --- |
| `vector_size` | 词向量、子词向量及输出向量的维度 | 基线为 64；用 128 检查表示容量、内存和耗时的变化 |
| `window` | 训练时的最大上下文距离 | 基线为 5；用 2 检查更局部的上下文 |
| `epochs` | 遍历语料的轮数及训练调度 | 基线为 5；用 10 检查增加优化轮数的影响，数据保持不变 |
| `min_count` | 从词表中排除低频完整词 | 基线为 2；用 5 检查词表缩小及 OOV 比例变化 |
| `min_n`、`max_n` | 字符子词的长度范围 | 基线为 2–4；用 3–6 改变子词粒度；`min_n=1, max_n=0` 关闭子词 |
| `bucket` | 子词哈希 ID 的取模范围 | 基线为 100000；用 10000 检查哈希冲突与内存的取舍 |
| `sg` | 词向量训练算法：`1` 为 skip-gram，`0` 为 CBOW | 基线为 skip-gram；将 CBOW 作为受控的架构对照 |
| `negative`、`hs` | 词向量训练目标 | 基线使用 5 个负样本和 `hs=0`；两者均不是监督分类的标签损失 |
| `sample`、`shrink_windows` | 高频词下采样与有效窗口的随机缩减 | 消融时保持 Gensim 默认值，以便隔离被比较的因素 |
| `workers`、`seed` | 线程调度与初始化 | 使用单 worker 和固定种子以减少波动；重复运行时设置 `PYTHONHASHSEED=0` |

## 评估边界

`run_experiments.py` 只用带弱标签的训练文章拟合 TF-IDF 和 LSI。Word2Vec、FastText 的无监督训练使用这些训练文章及未被选入分类任务的文章；带弱标签的验证、测试文章被排除。标题用于生成弱标签，但不作为分类器输入。`run_ablations.py` 读取同一份文章划分，每次只改变一组参数，并仅报告**验证集**指标，不用测试集结果挑选设置。这是基于一份随机划分和弱标签的探索性比较，不能据此断言某个设置在其他数据上普遍更好。

当前 Gensim 版本的 LSI 代码调用 `scipy.sparse.sparsetools.csc_matvecs`，因此需要低于 1.14 的 SciPy。`environment.yml` 和 `requirements.txt` 已固定这一兼容范围。
