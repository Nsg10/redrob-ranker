"""
rank.py — The Main Ranker

Runs in < 5 minutes, no GPU, no network calls.
Loads pre-computed cache and outputs top 100 candidates as CSV.

HOW TO RUN:
  python rank.py --cache ./cache --candidates ./candidates.jsonl --out ./submission.csv
"""

import argparse
import csv
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from utils.reasoning import generate_reasoning

WEIGHTS = {
    "semantic": 0.35,
    "career": 0.30,
    "skills": 0.25,
    "availability": 0.10,
}


def compute_final_score(semantic_score: float, feature_scores: dict) -> dict:
    career = feature_scores.get("career", {})
    skills = feature_scores.get("skills", {})
    behavioral = feature_scores.get("behavioral", {})
    honeypot = feature_scores.get("honeypot", {})

    career_score = career.get("career_score", 0.0)
    skill_score = skills.get("skill_score", 0.0)
    availability_score = behavioral.get("availability_score", 0.0)
    behavioral_multiplier = behavioral.get("behavioral_multiplier", 1.0)
    honeypot_penalty = honeypot.get("honeypot_penalty", 1.0)

    if career.get("disqualified", False):
        base_score = (
            WEIGHTS["semantic"] * semantic_score +
            WEIGHTS["career"] * career_score +
            WEIGHTS["skills"] * skill_score +
            WEIGHTS["availability"] * availability_score
        )
        final_score = base_score * 0.15
        return {
            "final_score": round(final_score, 6),
            "semantic_score": round(semantic_score, 4),
            "career_score": round(career_score, 4),
            "skill_score": round(skill_score, 4),
            "availability_score": round(availability_score, 4),
            "behavioral_multiplier": round(behavioral_multiplier, 3),
            "honeypot_penalty": honeypot_penalty,
            "disqualified": True,
        }

    base_score = (
        WEIGHTS["semantic"] * semantic_score +
        WEIGHTS["career"] * career_score +
        WEIGHTS["skills"] * skill_score +
        WEIGHTS["availability"] * availability_score
    )

    final_score = base_score * behavioral_multiplier * honeypot_penalty
    final_score = max(0.0, min(1.0, final_score))

    return {
        "final_score": round(final_score, 6),
        "semantic_score": round(semantic_score, 4),
        "career_score": round(career_score, 4),
        "skill_score": round(skill_score, 4),
        "availability_score": round(availability_score, 4),
        "behavioral_multiplier": round(behavioral_multiplier, 3),
        "honeypot_penalty": honeypot_penalty,
        "disqualified": False,
    }


def main():
    parser = argparse.ArgumentParser(description="Rank candidates against job description")
    parser.add_argument("--cache", default="./cache", help="Path to precomputed cache directory")
    parser.add_argument("--candidates", required=True, help="Path to candidates.jsonl")
    parser.add_argument("--out", default="./submission.csv", help="Output CSV path")
    parser.add_argument("--top-k", type=int, default=100, help="Number of candidates to output")
    args = parser.parse_args()

    print("=" * 55)
    print("  Redrob AI Hackathon — Intelligent Candidate Ranker")
    print("=" * 55)

    print("\n[1/5] Loading pre-computed cache...")
    jd_embedding = np.load(os.path.join(args.cache, "jd_embedding.npy"))
    candidate_embeddings = np.load(os.path.join(args.cache, "candidate_embeddings.npy"))

    with open(os.path.join(args.cache, "candidate_ids.pkl"), "rb") as f:
        candidate_ids = pickle.load(f)

    with open(os.path.join(args.cache, "feature_cache.pkl"), "rb") as f:
        feature_cache = pickle.load(f)

    print(f"  Embeddings: {candidate_embeddings.shape}")
    print(f"  Features: {len(feature_cache):,} candidates")

    print("\n[2/5] Computing semantic similarity scores...")
    semantic_scores = candidate_embeddings @ jd_embedding
    print(f"  Done | Mean: {semantic_scores.mean():.3f} | Max: {semantic_scores.max():.3f}")

    print("\n[3/5] Computing final composite scores...")
    all_scores = []
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}

    for cid in tqdm(candidate_ids, desc="  Scoring"):
        idx = id_to_idx[cid]
        sem_score = float(semantic_scores[idx])
        features = feature_cache.get(cid, {})
        score_result = compute_final_score(sem_score, features)
        score_result["candidate_id"] = cid
        score_result["feature_scores"] = features
        all_scores.append(score_result)

    all_scores.sort(key=lambda x: x["final_score"], reverse=True)

    top_100 = all_scores[:args.top_k]
    honeypot_count = sum(
        1 for s in top_100
        if s["feature_scores"].get("honeypot", {}).get("is_honeypot", False)
    )
    print(f"  Honeypots in top {args.top_k}: {honeypot_count} (limit: {args.top_k // 10})")

    print("\n[4/5] Loading candidate profiles for reasoning...")
    top_100_ids = {s["candidate_id"] for s in top_100}
    candidate_lookup = {}

    with open(args.candidates, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            if c["candidate_id"] in top_100_ids:
                candidate_lookup[c["candidate_id"]] = c
            if len(candidate_lookup) == len(top_100_ids):
                break

    print(f"  Loaded {len(candidate_lookup)} profiles")

    print(f"\n[5/5] Writing submission to {args.out}...")
    with open(args.out, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(
            csvfile,
            fieldnames=["candidate_id", "rank", "score", "reasoning"]
        )
        writer.writeheader()

        for rank_pos, score_data in enumerate(top_100, start=1):
            cid = score_data["candidate_id"]
            candidate = candidate_lookup.get(cid, {})

            reasoning = generate_reasoning(
                candidate=candidate,
                scores={
                    "career": score_data["feature_scores"].get("career", {}),
                    "skills": score_data["feature_scores"].get("skills", {}),
                    "behavioral": score_data["feature_scores"].get("behavioral", {}),
                    "semantic_score": score_data["semantic_score"],
                },
                rank=rank_pos
            )

            writer.writerow({
                "candidate_id": cid,
                "rank": rank_pos,
                "score": score_data["final_score"],
                "reasoning": reasoning,
            })

    print(f"  Submission written!")

    print("\n" + "=" * 55)
    print("  TOP 5 CANDIDATES")
    print("=" * 55)
    for i, s in enumerate(top_100[:5], 1):
        cid = s["candidate_id"]
        c = candidate_lookup.get(cid, {})
        p = c.get("profile", {})
        print(f"  #{i}: {p.get('current_title','?')} @ {p.get('current_company','?')}")
        print(f"       Score: {s['final_score']:.4f} | "
              f"sem={s['semantic_score']:.3f} | "
              f"career={s['career_score']:.3f} | "
              f"skills={s['skill_score']:.3f}")

    print(f"\n  Honeypots in top 100: {honeypot_count}/100")
    print(f"\n  Run: python validate_submission.py {args.out}")
    print("=" * 55)


if __name__ == "__main__":
    main()
