# 实验三：RAG 中文问答系统

## 实验目标
搭建一个基于 RAG（检索增强生成）技术的中文问答系统，
实现从文档检索到 LLM 生成的完整链路，并对比有无 RAG 的效果差异。

## 技术栈
| 组件 | 选型 |
|------|------|
| Embedding 模型 | text2vec-base-chinese（768维，中文优化）|
| 向量数据库 | FAISS IndexFlatIP |
| 大语言模型 | DeepSeek API (deepseek-chat) |
| 运行平台 | Google Colab |

## 完整流程

## 实验结论：有无 RAG 对比

| 对比维度 | 有 RAG | 无 RAG |
|----------|--------|--------|
| 回答依据 | 来自文档原文 | 依赖模型记忆 |
| 具体数值准确性 | 高（可溯源）| 可能有偏差 |
| 知识更新能力 | 改文档即更新 | 需重新训练 |
| 幻觉风险 | 低 | 较高 |

## 文件说明
- `rag_qa_experiment3.py` — 完整实验代码
- `rag_index.faiss` — FAISS 向量索引
- `rag_chunks.json` — chunk 原文存储

## 运行方式
1. 在 Google Colab 打开 `rag_qa_experiment3.py`
2. 安装依赖：`pip install faiss-cpu sentence-transformers openai langchain-text-splitters`
3. 替换 `DEEPSEEK_API_KEY` 为你的 Key
4. 顺序运行各 Cell
