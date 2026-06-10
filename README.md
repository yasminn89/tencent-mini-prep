# 腾讯 Mini 项目技术预研

> 深技大 × 腾讯 2026 Mini 项目实践记录  
> 参与项目：**项目2 RepoMind** · **项目3 性能 Skill 评估** · **项目5 元宝测试 Agent**

---

## 项目概览

本仓库记录参与腾讯 Mini 项目前的技术预研与实战实验，覆盖大模型微调、Prompt Engineering、RAG 检索增强生成三个核心方向。

| 实验 | 技术方向 | 对应 Mini 项目 | 状态 |
|------|----------|----------------|------|
| [BERT 中文情感分类](#实验1) | Fine-tuning · NLP | 项目2 RepoMind | 🔄 进行中 |
| [Prompt Engineering 文本评估](#实验2) | PE · Few-shot · CoT | 项目3 Skill 评估 | ⏳ 待开始 |
| [RAG 中文问答系统](#实验3) | RAG · 向量检索 · LLM | 项目2 / 3 / 5 | ⏳ 待开始 |

---

## 实验1：BERT 中文情感分类 Fine-tuning

### 实验目标
使用 `bert-base-chinese` 对中文酒店评论数据集 ChnSentiCorp 进行微调，实现二分类（好评 / 差评）。

### 技术栈
- **模型**：bert-base-chinese（HuggingFace）
- **框架**：PyTorch · HuggingFace Transformers · Datasets
- **平台**：Google Colab（T4 GPU）

### 与腾讯项目的关联
项目2 RepoMind 要求熟悉至少一种微调方法（LoRA/Fine-tuning）并掌握 HuggingFace Transformers、PEFT 框架。本实验完整覆盖 fine-tuning 全流程，为参与项目2奠定基础。

### 核心内容
- 数据集加载与预处理（tokenization、padding、truncation）
- BERT 分类头结构理解（BertForSequenceClassification）
- 训练参数配置（warmup、weight_decay、evaluation_strategy）
- 模型评估与推理（accuracy 指标）

---

## 实验2：Prompt Engineering 文本质量评估

### 实验目标
使用 LLM API 结合 System Prompt、Few-shot、Chain-of-Thought 技巧，构建一个可量化的文本输出质量评估器。

### 技术栈
- **模型**：主流 LLM API
- **技术**：System Prompt · Few-shot · CoT · 评分 Rubric 设计

### 与腾讯项目的关联
项目3 性能 Skill 评估的核心工作之一是「将模糊的业务需求转化为可量化评测指标」，本实验直接练习这一能力。

---

## 实验3：RAG 中文问答系统

### 实验目标
构建一个基于检索增强生成（RAG）的中文问答系统，集成向量数据库与 LLM。

### 技术栈
- **向量数据库**：FAISS / Chroma
- **Embedding 模型**：text2vec-base-chinese
- **框架**：LangChain · HuggingFace

### 与腾讯项目的关联
项目2 要求了解 RAG 架构与向量数据库；项目3 要求熟悉 FAISS/Milvus/Chroma；项目5 的测试用例生成也依赖 RAG 能力。本实验一次覆盖三个项目的共同技术需求。

---

## 目录结构

```
tencent-mini-prep/
├── experiment1-bert-finetune/
│   ├── bert_sentiment.ipynb    # 实验 notebook
│   └── README.md
├── experiment2-prompt-eval/
│   ├── prompt_eval.ipynb
│   └── README.md
├── experiment3-rag-qa/
│   ├── rag_qa.ipynb
│   └── README.md
└── README.md
```

---

## 环境配置

```bash
pip install transformers datasets evaluate torch
pip install langchain faiss-cpu chromadb
pip install openai
```

---

*持续更新中 · Last updated: 2026*
