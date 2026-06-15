"""
honeypot.py — Impossible Profile Detector

The dataset has ~80 honeypot candidates with subtly impossible profiles.
We catch them with timeline consistency checks.
Honeypot rate > 10% in top 100 = disqualification from the competition.
"""

from datetime import datetime

REFERENCE_DATE = datetime(2026, 6, 1)


def check_honeypot(candidate: dict) -> dict:
    profile = candidate.get("profile", {})
    career_history = candidate.get("career_history", [])
    skills = candidate.get("skills", [])

    reasons = []
    red_flags = 0

    stated_yoe = profile.get("years_of_experience", 0)
    total_career_months = sum(h.get("duration_months", 0) for h in career_history)
    total_career_years = total_career_months / 12.0

    if career_history:
        if total_career_years > stated_yoe + 3:
            reasons.append(
                f"Career history ({total_career_years:.1f}yr) far exceeds stated YoE ({stated_yoe}yr)"
            )
            red_flags += 2
        elif stated_yoe > total_career_years + 4:
            reasons.append(
                f"Stated YoE ({stated_yoe}yr) far exceeds career history ({total_career_years:.1f}yr)"
            )
            red_flags += 1

    for job in career_history:
        start_str = job.get("start_date", "")
        end_str = job.get("end_date", "")
        stated_duration = job.get("duration_months", 0)

        if not start_str:
            continue

        try:
            start = datetime.strptime(start_str, "%Y-%m-%d")
            end = datetime.strptime(end_str, "%Y-%m-%d") if end_str else REFERENCE_DATE
            actual_months = (end.year - start.year) * 12 + (end.month - start.month)

            if abs(actual_months - stated_duration) > 8:
                reasons.append(
                    f"Job at {job.get('company')}: stated {stated_duration}mo but dates show {actual_months}mo"
                )
                red_flags += 2

            if start > REFERENCE_DATE:
                reasons.append(f"Job at {job.get('company')} starts in the future: {start_str}")
                red_flags += 3

        except ValueError:
            red_flags += 0.5

    for skill in skills:
        skill_duration = skill.get("duration_months", 0) or 0
        grace_months = 36
        if skill_duration > total_career_months + grace_months:
            reasons.append(
                f"Skill '{skill['name']}' claimed for {skill_duration}mo but total career is {total_career_months:.0f}mo"
            )
            red_flags += 2

    expert_skills = [s for s in skills if s.get("proficiency") in ("expert", "advanced")]
    zero_endorsement_experts = [s for s in expert_skills if s.get("endorsements", 0) == 0]

    if len(expert_skills) >= 6 and len(zero_endorsement_experts) >= 5:
        reasons.append(
            f"{len(zero_endorsement_experts)} expert/advanced skills with 0 endorsements"
        )
        red_flags += 1.5

    education = candidate.get("education", [])
    if education and career_history:
        latest_grad_year = max(
            (e.get("end_year", 0) for e in education if e.get("end_year")),
            default=0
        )
        earliest_job_start = None
        for job in career_history:
            start_str = job.get("start_date", "")
            if start_str:
                try:
                    start_year = int(start_str[:4])
                    if earliest_job_start is None or start_year < earliest_job_start:
                        earliest_job_start = start_year
                except ValueError:
                    pass

        if latest_grad_year and earliest_job_start:
            if earliest_job_start < latest_grad_year - 2:
                reasons.append(
                    f"Started working ({earliest_job_start}) before graduating ({latest_grad_year})"
                )
                red_flags += 1.5

    is_honeypot = red_flags >= 4
    is_suspicious = 2 <= red_flags < 4

    if is_honeypot:
        honeypot_penalty = 0.05
    elif is_suspicious:
        honeypot_penalty = 0.60
    else:
        honeypot_penalty = 1.00

    return {
        "is_honeypot": is_honeypot,
        "is_suspicious": is_suspicious,
        "red_flags": red_flags,
        "honeypot_reasons": reasons,
        "honeypot_penalty": honeypot_penalty,
    }
