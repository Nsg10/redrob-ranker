"""
semantic.py — Multi-Section JD Semantic Scorer

Instead of embedding the whole JD as one blob,
we split it into 3 sections and score separately:

  1. Must-haves (60%) — core requirements
  2. Nice-to-haves (30%) — bonus signals  
  3. Anti-patterns (10% negative) — what we DON'T want

A candidate matching must-haves strongly but also matching
anti-patterns gets appropriately penalized.
"""

# Must-have requirements from JD
JD_MUST_HAVE = """
Production experience building embeddings-based retrieval systems.
Sentence transformers, text embeddings, dense retrieval, semantic search.
Vector databases: Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch, OpenSearch.
Hybrid search combining dense and sparse retrieval, BM25.
Strong Python engineering skills.
Evaluation frameworks for ranking: NDCG, MRR, MAP, A/B testing, offline evaluation.
Experience at product companies shipping real ML systems to real users.
5 to 9 years of hands-on applied ML engineering experience.
NLP and information retrieval background.
RAG, retrieval augmented generation, LLM integration.
Learning to rank, reranking, XGBoost, LightGBM ranking models.
Shipped production ML systems, not just research or prototypes.
"""

# Nice-to-have requirements from JD
JD_NICE_TO_HAVE = """
LLM fine-tuning experience: LoRA, QLoRA, PEFT, instruction tuning.
Distributed systems and inference optimization.
MLflow, Weights and Biases, experiment tracking.
Open source contributions in AI and ML.
Experience with both neural and traditional ranking approaches.
FastAPI, model serving, inference infrastructure.
Experience growing from individual contributor to tech lead.
"""

# Anti-patterns — what the JD explicitly says they do NOT want
JD_ANTI_PATTERNS = """
Consulting only career, worked only at TCS Infosys Wipro Accenture with no product company experience.
Computer vision only engineer with no NLP or text retrieval experience.
Speech recognition engineer with no information retrieval background.
Robotics engineer with no machine learning for text experience.
Pure researcher with no production deployment experience.
Recent LLM wrapper developer who only uses APIs without understanding retrieval systems.
Candidate who switches jobs every year with no depth in any role.
"""


def compute_multi_section_semantic(candidate_embedding, model):
    """
    Compute a multi-section semantic score.
    
    Args:
        candidate_embedding: normalized numpy array (384,)
        model: loaded SentenceTransformer model
    
    Returns:
        dict with combined semantic score and section scores
    """
    import numpy as np

    # Embed each JD section
    must_emb = model.encode(JD_MUST_HAVE, normalize_embeddings=True)
    nice_emb = model.encode(JD_NICE_TO_HAVE, normalize_embeddings=True)
    anti_emb = model.encode(JD_ANTI_PATTERNS, normalize_embeddings=True)

    # Cosine similarity for each section
    must_score = float(candidate_embedding @ must_emb)
    nice_score = float(candidate_embedding @ nice_emb)
    anti_score = float(candidate_embedding @ anti_emb)

    # Combined: must-haves dominate, nice-to-haves add, anti-patterns subtract
    combined = (
        0.60 * must_score +
        0.30 * nice_score -
        0.10 * anti_score
    )

    # Normalize to 0-1 range (combined can go slightly negative)
    combined = max(0.0, min(1.0, combined))

    return {
        "semantic_score": round(combined, 4),
        "must_have_score": round(must_score, 4),
        "nice_have_score": round(nice_score, 4),
        "anti_pattern_score": round(anti_score, 4),
    }


def get_jd_section_embeddings(model):
    """
    Pre-compute JD section embeddings once.
    Returns dict of section_name -> embedding array.
    """
    return {
        "must_have": model.encode(JD_MUST_HAVE, normalize_embeddings=True),
        "nice_have": model.encode(JD_NICE_TO_HAVE, normalize_embeddings=True),
        "anti_pattern": model.encode(JD_ANTI_PATTERNS, normalize_embeddings=True),
    }


def batch_compute_semantic_scores(candidate_embeddings, jd_sections):
    """
    Vectorized computation of multi-section semantic scores for all candidates.
    
    Args:
        candidate_embeddings: numpy array (N, 384)
        jd_sections: dict from get_jd_section_embeddings()
    
    Returns:
        numpy array (N,) of combined semantic scores
    """
    import numpy as np

    must_scores = candidate_embeddings @ jd_sections["must_have"]
    nice_scores = candidate_embeddings @ jd_sections["nice_have"]
    anti_scores = candidate_embeddings @ jd_sections["anti_pattern"]

    combined = (
        0.60 * must_scores +
        0.30 * nice_scores -
        0.10 * anti_scores
    )

    # Clip to 0-1
    return np.clip(combined, 0.0, 1.0)
