"""
behavioral.py — Behavioral Signals Scorer

Works as a MULTIPLIER on top of other scores.
A great candidate who hasn't logged in for 6 months gets scaled DOWN.
A highly active, responsive candidate gets a small boost.
"""

from datetime import datetime, date

PREFERRED_LOCATIONS = {
    "pune", "noida", "delhi", "delhi ncr", "gurgaon", "gurugram",
    "hyderabad", "mumbai", "bangalore", "bengaluru"
}

REFERENCE_DATE = date(2026, 6, 1)


def days_since(date_str: str) -> int:
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d").date()
        return (REFERENCE_DATE - d).days
    except Exception:
        return 365


def score_behavioral(candidate: dict) -> dict:
    sig = candidate.get("redrob_signals", {})
    profile = candidate.get("profile", {})

    last_active = sig.get("last_active_date", "")
    days_inactive = days_since(last_active) if last_active else 365

    if days_inactive <= 7:
        recency_score = 1.0
    elif days_inactive <= 30:
        recency_score = 0.9
    elif days_inactive <= 60:
        recency_score = 0.75
    elif days_inactive <= 90:
        recency_score = 0.55
    elif days_inactive <= 180:
        recency_score = 0.35
    else:
        recency_score = 0.15

    open_to_work = sig.get("open_to_work_flag", False)
    open_score = 1.0 if open_to_work else 0.5

    response_rate = sig.get("recruiter_response_rate", 0.0)
    avg_response_hours = sig.get("avg_response_time_hours", 999)

    if avg_response_hours <= 12:
        response_time_score = 1.0
    elif avg_response_hours <= 24:
        response_time_score = 0.85
    elif avg_response_hours <= 48:
        response_time_score = 0.65
    elif avg_response_hours <= 72:
        response_time_score = 0.45
    else:
        response_time_score = 0.25

    responsiveness_score = 0.65 * response_rate + 0.35 * response_time_score

    interview_rate = sig.get("interview_completion_rate", 0.5)
    offer_acceptance = sig.get("offer_acceptance_rate", -1)

    if offer_acceptance == -1:
        offer_score = 0.6
    else:
        offer_score = offer_acceptance

    reliability_score = 0.7 * interview_rate + 0.3 * offer_score

    notice_days = sig.get("notice_period_days", 60)

    if notice_days <= 15:
        notice_score = 1.0
    elif notice_days <= 30:
        notice_score = 0.9
    elif notice_days <= 60:
        notice_score = 0.7
    elif notice_days <= 90:
        notice_score = 0.45
    else:
        notice_score = 0.25

    location = profile.get("location", "").lower()
    country = profile.get("country", "").lower()
    willing_to_relocate = sig.get("willing_to_relocate", False)
    preferred_work_mode = sig.get("preferred_work_mode", "")

    in_preferred_city = any(city in location for city in PREFERRED_LOCATIONS)
    in_india = "india" in country or any(city in location for city in PREFERRED_LOCATIONS)

    if in_preferred_city:
        location_score = 1.0
    elif in_india and willing_to_relocate:
        location_score = 0.85
    elif in_india:
        location_score = 0.65
    elif willing_to_relocate:
        location_score = 0.50
    else:
        location_score = 0.20

    if preferred_work_mode == "remote" and not in_preferred_city:
        location_score *= 0.85

    saved_30d = sig.get("saved_by_recruiters_30d", 0)
    search_appear = sig.get("search_appearance_30d", 0)
    engagement_score = min((saved_30d * 3 + search_appear * 0.5) / 100.0, 1.0)

    availability_score = (
        0.25 * recency_score +
        0.20 * open_score +
        0.20 * responsiveness_score +
        0.15 * reliability_score +
        0.10 * notice_score +
        0.10 * engagement_score
    )

    if availability_score >= 0.80:
        behavioral_multiplier = 1.15
    elif availability_score >= 0.65:
        behavioral_multiplier = 1.05
    elif availability_score >= 0.50:
        behavioral_multiplier = 1.00
    elif availability_score >= 0.35:
        behavioral_multiplier = 0.80
    elif availability_score >= 0.20:
        behavioral_multiplier = 0.65
    else:
        behavioral_multiplier = 0.50

    return {
        "behavioral_multiplier": round(behavioral_multiplier, 3),
        "availability_score": round(availability_score, 4),
        "location_score": round(location_score, 4),
        "notice_score": round(notice_score, 4),
        "recency_score": round(recency_score, 4),
        "responsiveness_score": round(responsiveness_score, 4),
        "reliability_score": round(reliability_score, 4),
        "days_inactive": days_inactive,
        "notice_period_days": notice_days,
        "open_to_work": open_to_work,
    }
