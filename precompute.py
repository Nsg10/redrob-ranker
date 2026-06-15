"""
precompute.py — One-time Pre-computation Step

Run this ONCE before ranking. It:
1. Embeds the JD into a vector
2. Embeds all 100K candidate narratives
3. Pre-computes career/skills/behavioral scores
4. Saves everything to ./cache/

HOW TO RUN:
  python precompute.py --candidates ./candidates.jsonl --out ./cache
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from score.career import score_career
from score.skills import score_skills
from score.behavioral import score_behavioral
from utils.honeypot import check_honeypot


def build_candidate_text(candidate: dict) -> str:
    """
    Build a rich text narrative for embedding.
    We weight career descriptions heavily — they describe what
    the candidate actually DID, not just their job title.
    """
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    parts = []

    title = profile.get("current_title", "")
    company = profile.get("current_company", "")
    yoe = profile.get("years_of_experience", 0)
    if title:
        parts.append(f"Current role: {title} at {company} with {yoe} years experience.")

    summary = profile.get("summary", "")
    if summary:
        parts.append(summary[:500])

    for job in career_history[:4]:
        desc = job.get("description", "")
        job_title = job.get("title", "")
        job_company = job.get("company", "")
        if desc:
            parts.append(f"At {job_company} as {job_title}: {desc[:300]}")

    strong_skills = [
        s["name"] for s in skills
        if s.get("proficiency") in ("advanced", "expert", "intermediate")
        and s.get("duration_months", 0) >= 6
    ]
    if strong_skills:
        parts.append("Key skills: " + ", ".join(strong_skills[:12]))

    return " ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Pre-compute embeddings and features")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./cache", help="Output directory")
    parser.add_argument("--batch-size", type=int, default=512, help="Embedding batch size")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading sentence transformer model (all-MiniLM-L6-v2)...")
    print("  First run downloads ~80MB. Cached after that.")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print("  Model loaded")

    print("\nEmbedding job description...")
    jd_text = """
    Senior AI Engineer role requiring: production experience with embeddings-based retrieval systems
    (sentence-transformers, BGE, E5, OpenAI embeddings), vector databases (Pinecone, Weaviate, Qdrant,
    Milvus, FAISS, Elasticsearch), hybrid search infrastructure. Strong Python. Evaluation frameworks
    for ranking systems (NDCG, MRR, MAP, A/B testing). Experience at product companies (not consulting).
    5-9 years experience. Shipper mindset over researcher mindset. NLP and information retrieval background.
    LLM integration experience (fine-tuning, LoRA, RAG). Learning-to-rank models (XGBoost, neural LTR).
    Located in Pune, Noida, Hyderabad, Mumbai, Delhi NCR or willing to relocate.
    NOT: consulting-only career, computer vision/speech/robotics only, no production deployment,
    title-chaser, recent LLM-wrapper developer only.
    """
    jd_embedding = model.encode(jd_text, normalize_embeddings=True)
    np.save(os.path.join(args.out, "jd_embedding.npy"), jd_embedding)
    print(f"  JD embedded → shape {jd_embedding.shape}")

    print(f"\nLoading candidates from {args.candidates}...")
    candidates = []
    with open(args.candidates, "r") as f:
        for line in tqdm(f, desc="  Reading"):
            line = line.strip()
            if line:
                candidates.append(json.loads(line))
    print(f"  Loaded {len(candidates):,} candidates")

    print("\nComputing rule-based feature scores...")
    feature_cache = {}
    for c in tqdm(candidates, desc="  Scoring"):
        cid = c["candidate_id"]
        feature_cache[cid] = {
            "career": score_career(c),
            "skills": score_skills(c),
            "behavioral": score_behavioral(c),
            "honeypot": check_honeypot(c),
        }

    with open(os.path.join(args.out, "feature_cache.pkl"), "wb") as f:
        pickle.dump(feature_cache, f)
    print(f"  Features saved for {len(feature_cache):,} candidates")

    print(f"\nBuilding candidate texts and embedding {len(candidates):,} candidates...")
    print("  This takes 2-4 minutes on CPU. Go get a coffee.")
    candidate_ids = []
    candidate_texts = []

    for c in tqdm(candidates, desc="  Building texts"):
        candidate_ids.append(c["candidate_id"])
        candidate_texts.append(build_candidate_text(c))

    embeddings = model.encode(
        candidate_texts,
        batch_size=args.batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    np.save(os.path.join(args.out, "candidate_embeddings.npy"), embeddings)
    with open(os.path.join(args.out, "candidate_ids.pkl"), "wb") as f:
        pickle.dump(candidate_ids, f)

    print(f"  Embeddings saved → shape {embeddings.shape}")
    print("\n" + "="*50)
    print("Pre-computation complete!")
    print(f"Cache saved to: {args.out}/")
    print("\nNext: python rank.py --cache ./cache --candidates ./candidates.jsonl --out ./submission.csv")


if __name__ == "__main__":
    main()
