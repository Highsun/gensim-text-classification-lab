# Gensim 表示学习与文本分类实验

基于《人民日报》1998 年 1 月已分词语料，构建可复现的文本表示与分类流程。项目包含 Gensim 的词袋、TF-IDF、LSI、Word2Vec、FastText 表示，以及 scikit-learn 的线性分类器。当前只使用本地的 `data/199801.txt`；后续可替换或增加其他新闻语料。

## 数据与任务

请将原始语料放在 `data/199801.txt`。该文件留在本地，由 `.gitignore` 排除，不随代码提交。原始文件每行以 `日期-版面-文章-句子/词性` 开始，后续词项形如 `中国/ns`。`src/corpus.py` 按文章编号合并正文，首句作为标题。标点和纯数字不进入正文词序列。语料已经分词，因此默认流程无需重新用 jieba 分词；jieba 保留为新增原始中文语料时的依赖。

原始语料没有人工主题标签。`src/labels.py` 根据标题关键词生成 **弱标签**：`economy`、`sports`、`culture`、`science`、`international`。只保留命中单个主题且正文至少 30 词的文章。分类器只读取正文，不读取用于生成标签的标题。先用经济、体育、文化三类，再扩展到五类。弱标签会漏标或误标，指标仅衡量这个自动构造的任务，不能当作人工标注新闻分类的准确率。

划分按文章进行，随机种子固定为 42，训练/验证/测试比例为 60%/20%/20%，并分层抽样。TF-IDF 词表及 IDF、Gensim LSI 均只从分类训练集拟合。Word2Vec 和 FastText 在分类训练文章及未入选分类的文章上无监督训练，排除有标签的验证与测试文章。`outputs/default/splits.json` 保存文章级划分。

## 环境

在 `code/` 目录运行。建议创建位于项目内的独立 Conda 前缀环境：

```bash
mkdir -p .conda/pkgs .conda/cache
XDG_CACHE_HOME="$PWD/.conda/cache" CONDA_PKGS_DIRS="$PWD/.conda/pkgs" \
  conda env create -p "$PWD/.conda/env" -f environment.yml
conda activate "$PWD/.conda/env"
python -m pytest -q
PYTHONHASHSEED=0 python run_experiments.py
```

也可以用 `requirements.txt` 在独立虚拟环境中安装。`environment.yml` 与 `requirements.txt` 给出兼容版本范围；最终解析版本可用 `conda list -p .conda/env --explicit` 导出。所有生成的模型和指标放在 `outputs/`，环境与缓存放在 `.conda/`，均由 `.gitignore` 排除。无需向已有 `ai` 环境安装软件。清理时先退出该环境，再运行：

```bash
conda deactivate
conda env remove -p "$PWD/.conda/env"
rm -rf .conda outputs
```

## 运行与参数

```bash
python run_experiments.py --help
python run_experiments.py --output outputs/dim128 --vector-size 128 --epochs 10
python run_experiments.py --output outputs/subword3to6 --min-n 3 --max-n 6
python -m src.source_map
PYTHONHASHSEED=0 python run_ablations.py
PYTHONHASHSEED=0 python run_experiments.py --epochs 10 --output outputs/selected_epochs10
```

默认向量维度 64、窗口 5、最小词频 2、训练 5 轮、单 worker、FastText 字符 n-gram 长度 2–4、哈希桶 100000。`--workers 1` 有助于复现；增加 worker 可提速，但训练顺序可能导致数值差异。`--bucket` 增大可减少字符 n-gram 哈希冲突，也会增加内存。`--min-n/--max-n` 控制子词粒度；中文已分词语料宜比较多个设定。`--min-count` 增大可减少低频噪声，但降低词表覆盖。`--window` 调整上下文范围。`--epochs` 调整遍历次数，过高可能过拟合小语料。

`outputs/<run>/metrics.json` 含各类样本数、划分、词表、近邻词、OOV 向量可用性，以及每个表示在三类和五类任务上的验证与测试 accuracy、macro-F1 和逐类指标。`word2vec.model`、`fasttext.model` 可重新载入；输出不会提交到 Git。默认分类器为相同超参数的逻辑回归，以便比较表示。TF-IDF 是词频基线；Gensim LSI 将 TF-IDF 映射到低维主题空间；Word2Vec/FastText 对正文词向量取平均。这里的简单平均没有监督地调整词向量。

`run_ablations.py` 使用相同文章划分逐项改变子词开关、子词长度、哈希桶、维度、窗口、训练轮次、最低词频或 skip-gram/CBOW。输出 `outputs/ablations.json` 记录验证集 macro-F1、训练时间、词表、估计内存、桶占用与 OOV 可用性；不逐项查看测试集来挑选参数。`results/` 保存本次运行的轻量 JSON 指标快照，完整模型仍留在本地 `outputs/`。源码路径、论文差异和参数选择见[实现导览](docs/implementation-guide.md)。

## FastText 与论文及源码

Gensim `models.FastText` 对应 [Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) 的词表示学习：词向量由词自身及字符 n-gram 表示组成；子词哈希到固定桶，因而可为部分未登录词生成向量。实现入口见安装环境中的 `gensim/models/fasttext.py`：`FastText` 扩展 `Word2Vec`，`_do_train_job` 调用训练内核；`FastTextKeyedVectors.recalc_char_ngram_buckets` 构建子词索引，`get_vector` 处理词向量和 OOV。核心优化位于 `gensim/models/fasttext_inner.pyx`。`python -m src.source_map` 会输出当前安装版本的实际文件和行号。

源码阅读路径：`FastText.build_vocab` 和 `FastText.train` 继承自 `Word2Vec`，负责词表扫描、epoch 调度与学习率管理；`FastText._do_train_job` 将词序列交给 Cython 内核；`fasttext_inner.pyx` 中的 `compute_ngrams_bytes` 产生带边界符的 UTF-8 字符片段，训练例程聚合词与子词向量；`FastTextKeyedVectors.get_vector` 在推断时处理词表内词与 OOV。默认 `sg=1` 使用 skip-gram，`negative=5` 是 Gensim 默认负采样数。对比 `--min-n/--max-n/--bucket` 可观察子词覆盖与哈希桶的影响。

[Bag of Tricks for Efficient Text Classification](https://aclanthology.org/E17-2068/) 描述的是另一项监督任务：将词和词 n-gram 的表示聚合后直接训练分类输出层，并讨论层次 softmax。**本项目不是该论文分类模型的逐项复现**：Gensim 的 FastText 用于无监督词向量训练，分类层由 scikit-learn 逻辑回归承担；没有训练论文中的监督词 n-gram 表示或层次 softmax。这个区分可通过对照 [Gensim FastText API](https://radimrehurek.com/gensim/models/fasttext.html) 和上述论文核查。

| 概念 | 论文要点 | 本项目对应 |
| --- | --- | --- |
| 字符子词 | 词表示由字符 n-gram 参与构成 | `FastText` 的 `min_n`、`max_n`、`bucket` 与 `fasttext_inner.pyx` |
| 未登录词 | 可从已有子词构造向量 | `get_vector`；`metrics.json` 记录 OOV 可用性 |
| 文本分类 | 词与词 n-gram 聚合后监督训练分类层 | 预训练词向量均值 + 逻辑回归；未实现监督词 n-gram |
| 分类损失 | 论文讨论层次 softmax | 逻辑回归使用 scikit-learn 的分类优化 |

## 目录

```text
data/199801.txt          原始已分词语料
src/corpus.py            解析文章
src/labels.py            可审计的弱标签规则
src/source_map.py        定位已安装 Gensim 源码
run_experiments.py       训练、评估、保存结果
run_ablations.py         受控超参数对照（验证集）
docs/implementation-guide.md  论文、源码和参数导览
results/                已运行实验的 JSON 指标快照
tests/                   解析与标签规则测试
environment.yml         Conda 环境规格
requirements.txt         Python 依赖规格
```

## 后续扩展

新增语料时，先确认使用权限、编码、文档边界和标签来源，再增加对应加载器。若要评估真实主题分类，应使用人工标签或公开标注数据，并保持相同的文章级划分及无泄漏的训练流程。

## 许可

代码使用 [MIT License](LICENSE)。`data/199801.txt` 为课程提供的第三方语料，不受本仓库代码许可证覆盖；向公开 GitHub 仓库同步前需确认语料再分发权利。无权公开时，请先移除该文件，另提供数据获取说明。
