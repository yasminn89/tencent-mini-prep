# 实验三：RAG 中文问答系统
# 技术栈：text2vec-base-chinese + FAISS + DeepSeek API
# 平台：Google Colab

# ── 安装依赖（首次运行取消注释）──
# !pip install faiss-cpu sentence-transformers openai langchain-text-splitters langchain-core -q

import os, json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ══════════════════════════════════════════
# 1. 准备文档
# ══════════════════════════════════════════
documents = [
    """文档一：RAG（检索增强生成）技术介绍
    RAG（Retrieval-Augmented Generation）是一种结合了信息检索和大语言模型的技术框架。
    其核心思想是：在大语言模型回答问题之前，先从外部知识库中检索与问题最相关的内容，
    然后将检索结果和用户问题一起传给模型，让模型基于真实资料来回答。
    RAG 的主要优势：减少幻觉、知识库可随时更新、回答有据可查。
    典型应用：企业知识问答系统、技术文档查询、法律合规检索等。""",

    """文档二：向量数据库与 FAISS 介绍
    向量数据库是专门用来存储和检索向量（高维数值数组）的数据库。
    与传统数据库按关键词匹配不同，向量数据库通过计算向量之间的距离来找到语义相似的内容。
    FAISS（Facebook AI Similarity Search）是 Meta 开源的高效向量检索库。
    它支持百万级别的向量快速检索，是 RAG 系统中最常用的本地向量数据库之一。
    FAISS 支持多种索引类型：IndexFlatL2（精确检索）和 IndexIVFFlat（近似检索，更快）。""",

    """文档三：Embedding（嵌入）技术原理
    Embedding 是将文本转化为固定长度数值向量的技术。
    text2vec-base-chinese 是专门针对中文优化的 Sentence-BERT 模型，输出 768 维向量。
    语义相似的文本，其向量在高维空间中距离更近。
    使用专门为中文训练的模型，比通用多语言模型效果更好。""",

    """文档四：物联网（IoT）与 AI 结合的应用场景
    AIoT 典型应用：智能家居、工业预测性维护、智慧城市、农业 IoT。
    RAG 在 IoT 设备故障诊断中的应用：将设备手册和故障记录存入向量数据库，
    技术人员描述故障时系统自动检索最相关解决方案，降低对人工专家的依赖。"""
]

# ══════════════════════════════════════════
# 2. 文档切片
# ══════════════════════════════════════════
splitter = RecursiveCharacterTextSplitter(
    chunk_size=300, chunk_overlap=50,
    separators=["\n\n", "\n", "。", "；", "，", " ", ""]
)
all_chunks, chunk_sources = [], []
for doc_id, doc in enumerate(documents):
    for chunk in splitter.split_text(doc):
        all_chunks.append(chunk)
        chunk_sources.append(doc_id)
print(f"切片完成：{len(all_chunks)} 个 chunk")

# ══════════════════════════════════════════
# 3. Embedding 向量化
# ══════════════════════════════════════════
print("加载 text2vec-base-chinese 模型...")
embedding_model = SentenceTransformer("shibing624/text2vec-base-chinese")
chunk_embeddings = embedding_model.encode(
    all_chunks, batch_size=32,
    show_progress_bar=True, normalize_embeddings=True
)
print(f"向量化完成：{chunk_embeddings.shape}")

# ══════════════════════════════════════════
# 4. 存入 FAISS
# ══════════════════════════════════════════
dimension = chunk_embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(np.array(chunk_embeddings, dtype=np.float32))
faiss.write_index(faiss_index, "rag_index.faiss")
with open("rag_chunks.json", "w", encoding="utf-8") as f:
    json.dump({"chunks": all_chunks, "sources": chunk_sources}, f, ensure_ascii=False)
print(f"FAISS 索引构建完成：{faiss_index.ntotal} 个向量")

# ══════════════════════════════════════════
# 5. 配置 DeepSeek API
# ══════════════════════════════════════════
DEEPSEEK_API_KEY = "your-api-key-here"  # 替换为你的 Key
client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")

# ══════════════════════════════════════════
# 6. RAG 核心函数
# ══════════════════════════════════════════
def retrieve(query, top_k=3):
    query_vec = embedding_model.encode([query], normalize_embeddings=True).astype(np.float32)
    scores, indices = faiss_index.search(query_vec, top_k)
    return [{"text": all_chunks[idx], "score": float(s), "source": chunk_sources[idx]}
            for s, idx in zip(scores[0], indices[0]) if idx != -1]

def rag_answer(query, top_k=3):
    retrieved = retrieve(query, top_k)
    context = "\n\n".join([f"【片段{i+1}】\n{r['text']}" for i, r in enumerate(retrieved)])
    prompt = f"""根据以下参考资料回答问题，资料不足时如实说明。

=== 参考资料 ===
{context}

=== 问题 ===
{query}

=== 回答 ==="""
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "你是专业技术问答助手，基于参考资料回答。"},
                  {"role": "user", "content": prompt}],
        max_tokens=600, temperature=0.1
    )
    return resp.choices[0].message.content

def direct_llm_answer(query):
    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "system", "content": "你是专业技术问答助手。"},
                  {"role": "user", "content": query}],
        max_tokens=400, temperature=0.1
    )
    return resp.choices[0].message.content

# ══════════════════════════════════════════
# 7. 效果对比测试
# ══════════════════════════════════════════
questions = [
    "text2vec-base-chinese 输出多少维的向量？",
    "FAISS 的 IndexFlatIP 和 IndexFlatL2 有什么区别？",
    "RAG 技术在物联网设备故障诊断中怎么应用？",
]

for q in questions:
    print(f"\n{'='*50}")
    print(f"问题：{q}")
    print(f"\n【有 RAG】")
    print(rag_answer(q))
    print(f"\n【无 RAG】")
    print(direct_llm_answer(q))
