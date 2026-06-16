# Redrob AI Hackathon — Intelligent Candidate Ranker

India Runs Hackathon | Track 1: Data & AI Challenge

A multi-signal candidate ranking system that ranks 100,000 candidates against a Senior AI Engineer job description using semantic embeddings + career quality scoring + behavioral signals.

## Architecture

The system runs in two phases:

**Phase 1 — precompute.py (run once, no time limit)**
- Embeds the JD into 3 sections: must-have, nice-to-have, anti-patterns
- Embeds all 100K candidate career narratives using all-MiniLM-L6-v2
- Pre-computes career quality, skills trust, behavioral, and honeypot scores
- Saves everything to ./cache/

**Phase 2 — rank.py (< 5 minutes, CPU only, no network)**
- Loads pre-computed cache
- Vectorized semantic similarity via dot product
- Combines 4 signals into weighted composite score
- Applies behavioral multiplier and honeypot penalty
- Outputs top 100 candidates as CSV with reasoning

## Scoring Weights

| Signal | Weight | What it measures |
|--------|--------|-----------------|
| Semantic similarity | 35% | Career narrative vs JD meaning (3-section embeddings) |
| Career quality | 30% | Product companies, domain fit, production experience |
| Skills trust | 25% | Skills × proficiency × duration × endorsements |
| Availability | 10% | Active on platform, open to work |
| Behavioral multiplier | ×0.5–1.15 | All 23 Redrob signals |
| Honeypot penalty | ×0.05–1.0 | Timeline consistency check |

## Hard Disqualifiers

- Consulting-only career (TCS, Infosys, Wipro, Accenture, etc.)
- Wrong domain only (CV/speech/robotics without any NLP/IR)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage

### Step 1: Pre-compute (run once)
```bash
python precompute.py --candidates ./candidates.jsonl --out ./cache
```

### Step 2: Rank
```bash
python rank.py --cache ./cache --candidates ./candidates.jsonl --out ./submission.csv
```

### Step 3: Validate
```bash
python validate_submission.py submission.csv
```

### Step 4: Demo (Streamlit sandbox)
```bash
streamlit run app.py
```

## File Structure
redrob-ranker/

├── precompute.py          # One-time embedding + feature computation

├── rank.py                # Main ranker (< 5 min, no GPU, no network)

├── app.py                 # Streamlit demo sandbox

├── score/

│   ├── career.py          # Career quality + disqualifier detection

│   ├── skills.py          # Trust-weighted skill scoring

│   ├── semantic.py        # 3-section JD semantic scorer

│   └── behavioral.py      # All 23 Redrob platform signals

├── utils/

│   ├── honeypot.py        # Impossible profile detection

│   └── reasoning.py       # Per-candidate reasoning generation

├── requirements.txt

└── submission.csv         # Final ranked output (top 100)

## Live Sandbox

https://redrob-ranker-5uekmfroven3nz82q4pven.streamlit.app/

## Results

- Top candidate: Staff ML Engineer @ Yellow.ai (score: 0.9356)
- 0 honeypots in top 100
- 6 hard-disqualified consulting-only candidates correctly pushed to bottom
- Ranking completes in < 5 seconds after precompute
