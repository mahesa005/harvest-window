#!/usr/bin/env python3
"""
HarvestWindow — Rule Engine

Pure logic, no ML dependency. Implements the locked B-stage x ripeness
recommendation table exactly as designed:

  - B1/B2 + any ripeness -> real recommendation (harvest now/immediately,
    or a recheck window)
  - B3/B4 + anything past "unripe" -> anomaly (implausible combination,
    the model's own internal contradiction, not a legitimate edge case)

Also computes: return-visit window (the core MVP selling point),
confidence display value, cost/ROI estimate, and the plain-language
reasoning string. This is exactly what Component/Flowchart Level 3/5
in the system design doc call "Rule engine" — everything from
"Combine classifications" through "Generate reasoning text" lives here
as one deterministic, unit-testable module.

Fully testable right now with made-up inputs — no trained weights needed.
"""

from dataclasses import dataclass
from typing import Optional

STAGES = ["B1", "B2", "B3", "B4"]
RIPENESS_LEVELS = ["unripe", "partially_ripe", "ripe", "overripe", "decayed"]


@dataclass
class Recommendation:
    status: str                      # "classified" | "anomaly"
    action: str                      # "harvest_now" | "harvest_immediately" | "recheck" | "anomaly"
    recommendation_text: str
    recheck_window_weeks: Optional[list]   # [min, max] or None
    reasoning: str


# The locked table. Keys: (stage, ripeness) -> (action, window_weeks_or_None, reasoning)
RECOMMENDATION_TABLE = {
    ("B1", "unripe"):         ("recheck", [1, 3], "Bunch stage indicates near-term readiness, not yet ripe"),
    ("B1", "partially_ripe"): ("recheck", [1, 2], "Ripening ahead of a typical unripe reading at this stage"),
    ("B1", "ripe"):           ("harvest_now", None, "Color shows ripe, bunch stage confirms readiness"),
    ("B1", "overripe"):       ("harvest_immediately", None, "Overripe at near-harvest stage — value being lost"),
    ("B1", "decayed"):        ("harvest_immediately", None, "Decayed at near-harvest stage — quality loss already occurred"),

    ("B2", "unripe"):         ("recheck", [4, 6], "Bunch stage indicates roughly 2 months out, not yet ready"),
    ("B2", "partially_ripe"): ("recheck", [3, 4], "Ripening slightly ahead of the bunch stage estimate"),
    ("B2", "ripe"):           ("harvest_now", None, "Color shows ripe, ahead of the typical stage timeline"),
    ("B2", "overripe"):       ("anomaly", None, "Overripe at a ~2-month-out stage — implausible, verify manually"),
    ("B2", "decayed"):        ("anomaly", None, "Decayed at a ~2-month-out stage — implausible, verify manually"),

    ("B3", "unripe"):         ("recheck", [6, 9], "Bunch stage indicates roughly 3 months out, not yet ready"),
    ("B3", "partially_ripe"): ("anomaly", None, "Partially ripe at a ~3-month-out stage — implausible, verify manually"),
    ("B3", "ripe"):           ("anomaly", None, "Ripe at a ~3-month-out stage — implausible, verify manually"),
    ("B3", "overripe"):       ("anomaly", None, "Overripe at a ~3-month-out stage — implausible, verify manually"),
    ("B3", "decayed"):        ("anomaly", None, "Decayed at a ~3-month-out stage — implausible, verify manually"),

    ("B4", "unripe"):         ("recheck", [10, 13], "Bunch stage indicates roughly 4 months out, not yet ready"),
    ("B4", "partially_ripe"): ("anomaly", None, "Partially ripe at a ~4-month-out stage — implausible, verify manually"),
    ("B4", "ripe"):           ("anomaly", None, "Ripe at a ~4-month-out stage — implausible, verify manually"),
    ("B4", "overripe"):       ("anomaly", None, "Overripe at a ~4-month-out stage — implausible, verify manually"),
    ("B4", "decayed"):        ("anomaly", None, "Decayed at a ~4-month-out stage — implausible, verify manually"),
}

ACTION_TEXT = {
    "harvest_now": "Harvest now",
    "harvest_immediately": "Harvest immediately — value being lost",
}


def compute_recommendation(stage: str, ripeness: str) -> Recommendation:
    """
    Core lookup. Raises ValueError on invalid stage/ripeness rather than
    silently defaulting — a typo here should crash loudly, not produce
    a wrong recommendation to a real harvester.
    """
    if stage not in STAGES:
        raise ValueError(f"Unknown stage '{stage}' — expected one of {STAGES}")
    if ripeness not in RIPENESS_LEVELS:
        raise ValueError(f"Unknown ripeness '{ripeness}' — expected one of {RIPENESS_LEVELS}")

    action, window, reasoning = RECOMMENDATION_TABLE[(stage, ripeness)]

    if action == "anomaly":
        return Recommendation(
            status="anomaly",
            action="anomaly",
            recommendation_text="Verify manually",
            recheck_window_weeks=None,
            reasoning=reasoning,
        )

    if action == "recheck":
        text = f"Come back in about {window[0]}\u2013{window[1]} weeks"
    else:
        text = ACTION_TEXT[action]

    return Recommendation(
        status="classified",
        action=action,
        recommendation_text=text,
        recheck_window_weeks=window,
        reasoning=reasoning,
    )


def compute_confidence(model1_conf: float, model2_conf: float) -> int:
    """
    Displayed as "agrees with expert grading at X%" per NFR9's framing
    constraint. Simple average for now — revisit if backtest results
    suggest one model's confidence is more reliable than the other's.
    """
    return round((model1_conf + model2_conf) / 2 * 100)


def compute_savings_idr(stage: str, ripeness: str, base_daily_cost: float = 15000) -> int:
    """
    Placeholder cost/ROI logic — FR9. Longer recheck windows imply more
    avoided low-value trips; harvest-now/immediately implies avoided
    quality-loss cost instead. Numbers here are illustrative, not
    sourced — same caveat as the recheck-window ranges themselves.
    """
    action, window, _ = RECOMMENDATION_TABLE[(stage, ripeness)]
    if action == "harvest_immediately":
        return int(base_daily_cost * 3)   # avoided further quality loss
    if action == "harvest_now":
        return int(base_daily_cost * 1)
    if action == "recheck" and window:
        avg_weeks = sum(window) / 2
        return int(base_daily_cost * avg_weeks * 0.5)   # avoided premature/wasted trip cost
    return 0


if __name__ == "__main__":
    # Smoke test — every cell in the table, no trained model needed at all
    print(f"{'Stage':<5} {'Ripeness':<15} {'Action':<20} {'Window':<10} Text")
    print("-" * 80)
    for stage in STAGES:
        for ripeness in RIPENESS_LEVELS:
            rec = compute_recommendation(stage, ripeness)
            window_str = str(rec.recheck_window_weeks) if rec.recheck_window_weeks else "-"
            print(f"{stage:<5} {ripeness:<15} {rec.action:<20} {window_str:<10} {rec.recommendation_text}")

    print("\n--- Sample full outputs ---")
    for stage, ripeness in [("B1", "ripe"), ("B2", "unripe"), ("B3", "overripe")]:
        rec = compute_recommendation(stage, ripeness)
        conf = compute_confidence(0.91, 0.85)
        savings = compute_savings_idr(stage, ripeness)
        print(f"\n{stage} + {ripeness}:")
        print(f"  status: {rec.status}")
        print(f"  recommendation: {rec.recommendation_text}")
        print(f"  recheck_window_weeks: {rec.recheck_window_weeks}")
        print(f"  confidence: {conf}")
        print(f"  estimated_savings_idr: {savings}")
        print(f"  reasoning: {rec.reasoning}")
