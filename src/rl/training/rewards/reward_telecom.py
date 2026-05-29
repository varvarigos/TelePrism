"""
Reward function for TelecomTS QA dataset with 8 task types.

Model output format (cold-start trained):
  <think>...reasoning...</think>
  ...bullet points...
  Therefore, this pattern is consistent with 'answer'.
  (anomaly_bounds: "Therefore, the anomaly occurred between indices 'X, Y'")

Answer extraction: strip <think> block, get last sentence, extract 'xxx' from quotes.

Task answers:
  zone:              'zone a', 'zone b', 'zone c'
  activity:          'youtube', 'twitch', 'file'
  anomaly_detection: 'yes anomaly detected', 'no anomaly detected'
  motion:            'mobile', 'stationary'
  cong:              'Congestion', 'No Congestion'
  root_cause:        e.g. 'co-channel interference (mild)', ...
  anomaly_bounds:    'X, Y'  (start, end indices)
  jam:               'yes', 'no'

Reward structure:
  score = correctness_weight * task_score + reasoning_weight * reasoning_score
          + format_bonus
  where:
    correctness_weight = 0.50 (binary per-task correctness)
    reasoning_weight   = 0.40 (continuous KPI-based reasoning quality)
    format_bonus       = 0.10 max (0.05 think tags + 0.05 quoted answer)

  The reasoning component provides continuous signal (0.0-1.0) based on
  whether the model identifies discriminating KPIs for the ground-truth class.
  This turns the reward from near-binary {0.1, 1.0} into a richer continuous
  distribution, reducing zero-variance GRPO groups and providing meaningful
  gradient signal to the 500+ reasoning tokens.
"""

import re
import string
from typing import Dict, List, Optional

# Import anomaly KPI mappings for reasoning bonus scoring
from teleprism.dataset.qa_templates import (
    anomalies_type_2_id,
    anomaly_type_2_affected_kpis,
)

# KPI name aliases for reasoning bonus
_KPI_ALIASES: Dict[str, List[str]] = {
    "RSRP": ["rsrp", "reference signal", "signal strength", "signal power"],
    "DL_BLER": ["dl_bler", "dl bler", "downlink bler", "downlink block error"],
    "UL_BLER": ["ul_bler", "ul bler", "uplink bler", "uplink block error"],
    "DL_MCS": ["dl_mcs", "dl mcs", "downlink mcs", "downlink modulation"],
    "UL_MCS": ["ul_mcs", "ul mcs", "uplink mcs", "uplink modulation"],
    "UL_SNR": ["ul_snr", "ul snr", "uplink snr", "signal to noise", "signal-to-noise"],
    "UL_NPRB": ["ul_nprb", "ul nprb", "uplink nprb", "uplink prb allocation"],
    "TX_Bytes": ["tx_bytes", "tx bytes", "transmitted bytes", "transmit bytes"],
    "RX_Bytes": ["rx_bytes", "rx bytes", "received bytes", "receive bytes"],
    "Estimated_UL_Buffer": ["ul_buffer", "ul buffer", "uplink buffer", "buffer backlog", "buffer overflow"],
    "PRBs_DL_Current": ["prbs_dl", "prb_dl", "downlink prb", "dl prb"],
    "PRBs_UL_Current": ["prbs_ul", "prb_ul", "uplink prb", "ul prb"],
    "PRB_Utilization_DL": ["prb_utilization_dl", "dl utilization", "downlink utilization", "dl prb utilization"],
    "PRB_Utilization_UL": ["prb_utilization_ul", "ul utilization", "uplink utilization", "ul prb utilization"],
    "UL_NumberOfPackets": ["ul_numberofpackets", "ul packets", "uplink packets", "uplink packet count"],
    "DL_NumberOfPackets": ["dl_numberofpackets", "dl packets", "downlink packets", "downlink packet count"],
}


# Helpers
def _strip_think(text: str) -> str:
    """Remove <think>...</think> block from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()



def extract_solution(solution_str: str) -> str:
    """Extract the answer from the model's output.

    Expected format:
      <think>...reasoning...</think>
      ...bullet points...
      Therefore, this pattern is consistent with 'answer'.

    Strategy: strip <think> block, get last sentence, extract 'xxx' from quotes.
    """
    solution_str = solution_str.replace("<|endoftext|>", "")
    solution_str = solution_str.replace("<|im_end|>", "")

    # Remove <think>...</think>
    stripped = _strip_think(solution_str)
    if not stripped:
        return ""

    # Get last sentence (split on period+space, newline, etc.)
    sentences = re.split(r'(?<=[.!?])\s+|\n+', stripped)
    last_sentence = ""
    for s in reversed(sentences):
        s = s.strip()
        if s:
            last_sentence = s
            break

    if not last_sentence:
        return stripped.strip()

    # Extract content in single quotes from last sentence
    matches = re.findall(r"'([^']+)'", last_sentence)
    if matches:
        # Return the last quoted string (the answer)
        return matches[-1].strip()

    # Fallback: return whole last sentence
    return last_sentence.strip()


def is_refusal(text: str) -> bool:
    """Detect if the model refused to answer."""
    t = text.lower()
    refusal_patterns = [
        "cannot determine", "can't determine", "unable to determine",
        "insufficient information", "not enough information",
        "i don't know", "i do not know",
    ]
    return any(p in t for p in refusal_patterns)


def is_gibberish(text: str) -> bool:
    """Detect if text is repetitive gibberish."""
    if len(text) < 20:
        return False
    # Check for repeated 2-5 char sequences taking up >50% of text
    for n in range(2, 6):
        if len(text) < n * 4:
            continue
        # Check last chunk for repetition
        chunk = text[-200:] if len(text) > 200 else text
        # Split into n-grams
        ngrams = [chunk[i:i+n] for i in range(0, len(chunk) - n + 1, n)]
        if ngrams:
            from collections import Counter
            most_common_count = Counter(ngrams).most_common(1)[0][1]
            if most_common_count / len(ngrams) > 0.5:
                return True
    return False


def count_kpi_mentions(text: str, kpi_list: List[str]) -> int:
    """Count how many KPIs from kpi_list are mentioned in text."""
    t = text.lower()
    count = 0
    for kpi in kpi_list:
        aliases = _KPI_ALIASES.get(kpi, [kpi.lower()])
        if any(alias in t for alias in aliases):
            count += 1
    return count


# Scoring functions — Binary tasks
def score_anomaly_detection(predicted: str, ground_truth: str) -> float:
    """Binary: 'yes anomaly detected' / 'no anomaly detected'."""
    p = predicted.lower().strip()
    g = ground_truth.lower().strip()

    def classify(text):
        if "no anomaly" in text or text == "no":
            return "no"
        if "yes anomaly" in text or "anomaly detected" in text or text == "yes":
            return "yes"
        if text.startswith("no"):
            return "no"
        if text.startswith("yes"):
            return "yes"
        return None

    pc, gc = classify(p), classify(g)
    if pc is None or gc is None:
        return 0.0
    return 1.0 if pc == gc else 0.0


def score_motion(predicted: str, ground_truth: str) -> float:
    """Binary: 'stationary' / 'mobile'."""
    p = predicted.lower().strip()
    g = ground_truth.lower().strip()

    def classify(text):
        if "stationary" in text or "static" in text:
            return "stationary"
        if "mobile" in text or "moving" in text or "in motion" in text:
            return "mobile"
        return None

    pc, gc = classify(p), classify(g)
    if pc is None or gc is None:
        return 0.0
    return 1.0 if pc == gc else 0.0


def score_jam(predicted: str, ground_truth: str) -> float:
    """Binary: jammer present / absent."""
    p = predicted.lower().strip()
    g = ground_truth.lower().strip()

    def classify(text):
        if "no jammer" in text or "no jamming" in text or text == "no":
            return "no"
        if "jammer" in text or "jamming" in text or text == "yes":
            return "yes"
        if text.startswith("no"):
            return "no"
        if text.startswith("yes"):
            return "yes"
        return None

    pc, gc = classify(p), classify(g)
    if pc is None or gc is None:
        return 0.0
    return 1.0 if pc == gc else 0.0


# Scoring functions — Categorical tasks
def score_zone(predicted: str, ground_truth: str) -> float:
    """3-class: 'zone a' / 'zone b' / 'zone c'."""
    # NOTE: do NOT use normalize_answer here — it strips article "a"
    p = predicted.lower().strip()
    g = ground_truth.lower().strip()

    def classify(text):
        if "zone a" in text or "zone 1" in text:
            return "a"
        if "zone b" in text or "zone 2" in text:
            return "b"
        if "zone c" in text or "zone 3" in text:
            return "c"
        # Single letter fallback
        text_clean = text.strip().rstrip(".")
        if text_clean in ("a", "b", "c"):
            return text_clean
        return None

    pc, gc = classify(p), classify(g)
    if pc is None or gc is None:
        return 0.0
    return 1.0 if pc == gc else 0.0


def score_activity(predicted: str, ground_truth: str) -> float:
    """3-class: 'youtube' / 'twitch' / 'file'."""
    p = predicted.lower().strip()
    g = ground_truth.lower().strip()

    def classify(text):
        if "youtube" in text or "yt" in text:
            return "youtube"
        if "twitch" in text:
            return "twitch"
        # "file" must be checked carefully — "file download" contains "file"
        if "file" in text:
            return "file"
        if "download" in text:
            return "file"
        return None

    pc, gc = classify(p), classify(g)
    if pc is None or gc is None:
        return 0.0
    return 1.0 if pc == gc else 0.0


# Scoring with reasoning bonus
def score_root_cause(predicted: str, ground_truth: str,
                     full_response: str = "") -> float:
    """11-class anomaly type.

    Both prediction and ground truth are denormalized to a canonical anomaly
    type using the synonym dict.  Returns 1.0 for a correct match, 0.0
    otherwise.  The full_response parameter is kept for API compatibility
    but is no longer used.
    """
    anomaly_synonyms: Dict[str, List[str]] = {
        "antenna failure": ["antenna failure", "antenna"],
        "co-channel interference (mild)": [
            "co-channel interference (mild)", "cochannel interference mild",
            "mild interference",
        ],
        "co-channel interference (severe)": [
            "co-channel interference (severe)", "cochannel interference severe",
            "severe interference",
        ],
        "faulty rf filters (temporal)": [
            "faulty rf filters (temporal)", "faulty rf filter", "rf filter",
        ],
        "doppler shift (severe)": [
            "doppler shift (severe)", "doppler shift", "doppler",
        ],
        "faulty handover algorithm (too frequent)": [
            "faulty handover algorithm (too frequent)", "faulty handover",
            "handover algorithm",
        ],
        "buffer overflow (gradual buildup)": [
            "buffer overflow (gradual buildup)", "buffer overflow",
        ],
        "resource allocation bugs": [
            "resource allocation bugs", "resource allocation bug",
            "resource allocation",
        ],
        "high network congestion (gradual buildup)": [
            "high network congestion (gradual buildup)",
            "network congestion gradual", "gradual congestion",
        ],
        "high network congestion (sudden spike)": [
            "high network congestion (sudden spike)",
            "network congestion sudden", "sudden congestion",
        ],
        "jamming": ["jamming", "jammer"],
    }

    p = predicted.lower().strip()
    g = ground_truth.lower().strip()

    def normalize(text: str) -> Optional[str]:
        """Map raw text → canonical anomaly type via synonym matching."""
        for canonical, keywords in anomaly_synonyms.items():
            if any(kw in text for kw in keywords):
                return canonical
        return None

    pc = normalize(p)
    gc = normalize(g)

    if gc is None:
        # Unknown GT: fall back to exact string match
        return 1.0 if p == g else 0.0

    if pc is None:
        return 0.0

    return 1.0 if pc == gc else 0.0


def score_cong(predicted: str, ground_truth: str,
               full_response: str = "") -> float:
    """Binary congestion — pure accuracy.

    Cold-start answers: 'Congestion' or 'No Congestion'.
    Returns 1.0 for correct, 0.0 for wrong.
    The full_response parameter is kept for API compatibility but is no
    longer used.
    """
    p = predicted.lower().strip()
    g = ground_truth.lower().strip()

    def classify(text):
        if "no congestion" in text or "not congested" in text:
            return "no"
        if "congestion" in text or "congested" in text:
            return "yes"
        if text.startswith("no"):
            return "no"
        if text.startswith("yes"):
            return "yes"
        return None

    pc, gc = classify(p), classify(g)
    if pc is None or gc is None:
        return 0.0
    return 1.0 if pc == gc else 0.0


# Anomaly bounds (IoU-based)
def score_anomaly_bounds(predicted: str, ground_truth: str) -> float:
    """Temporal IoU for anomaly start/end timestamps.

    Expected format: 'X, Y' (e.g. '8, 127').
    """

    def extract_pair(text: str):
        t = text.strip().strip("'\"")
        # Pattern 1: "X, Y" (direct from quotes)
        m = re.match(r"(\d+)\s*,\s*(\d+)", t)
        if m:
            return int(m.group(1)), int(m.group(2))
        # Pattern 2: "starts at X ... ends at Y"
        m = re.search(r"starts?\s+at\s+(\d+).*?ends?\s+at\s+(\d+)", t, re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
        # Pattern 3: "start: X end: Y"
        m = re.search(r"start\s*:?\s*(\d+).*?end\s*:?\s*(\d+)", t, re.IGNORECASE)
        if m:
            return int(m.group(1)), int(m.group(2))
        # Pattern 4: just two numbers anywhere
        nums = re.findall(r"\d+", t)
        if len(nums) >= 2:
            return int(nums[0]), int(nums[1])
        return None, None

    pred_s, pred_e = extract_pair(predicted)
    truth_s, truth_e = extract_pair(ground_truth)

    if pred_s is None or pred_e is None:
        return 0.0
    if truth_s is None or truth_e is None:
        return 0.0

    # Ensure start < end
    if pred_s > pred_e:
        pred_s, pred_e = pred_e, pred_s
    if truth_s > truth_e:
        truth_s, truth_e = truth_e, truth_s

    # Exact match
    if pred_s == truth_s and pred_e == truth_e:
        return 1.0

    # IoU
    inter_start = max(pred_s, truth_s)
    inter_end = min(pred_e, truth_e)

    if inter_start >= inter_end:
        return 0.0

    intersection = inter_end - inter_start
    union = max(pred_e, truth_e) - min(pred_s, truth_s)
    iou = intersection / union if union > 0 else 0.0
    return max(0.0, min(1.0, iou))


# ── Zone reasoning quality reward ─────────────────────────────────
# Derived from statistical analysis of 3954 training samples:
#
# Time-series data (18 channels, 128 steps):
#   Zone A: RSRP mean=-93 (std=7.8), range [-105, -82]
#   Zone B: RSRP mean=-110 (std=4.4), range [-115, -102]
#   Zone C: RSRP mean=-114 (std=3.8), range [-118, -105]
#   All other KPIs have Cohen's d < 0.31 between B and C.
#   RSRP is the ONLY strong discriminator (d=1.18 for B vs C).
#
# SFT CoT keyword analysis (200 samples each):
#   Zone A: "strong" 93%, "proximity/close" 75%, RSRP > -95
#   Zone B: "moderate" 88%, "close" 91%, RSRP -100 to -114
#           Key: model says "moderate" NOT "weak", "close" NOT "far"
#   Zone C: "weak" 100%, "far/distance" 51%, "compensate" 32%
#           Key: model says "weak" and "far"
#
# The reward uses TWO components:
#   1. Positive: credit for patterns that match the GT zone
#   2. Negative: penalty for patterns that match a DIFFERENT zone
# This prevents the model from being rewarded for zone C reasoning
# when the GT is zone B (the core failure mode).

# Patterns that are EXCLUSIVE to each zone (reward if present for GT zone,
# penalize if present for wrong zone)
_ZONE_POSITIVE: Dict[str, List[tuple]] = {
    "zone a": [
        # RSRP specifically described as strong (tight patterns)
        (r"(?:strong|high)\s+(?:rsrp|signal\s+strength)", 0.20),
        (r"(?:rsrp|signal\s+strength)\s+(?:is\s+)?(?:strong|high|good)", 0.10),
        # RSRP > -105 dB: captures 90% of zone A, only 18% of B, 3% of C
        (r"-(?:8\d|9\d|10[0-4])\.?\d*\s*d[Bb]", 0.25),
        # Proximity: "close/near base station"
        (r"(?:proxim|close|near).*?(?:base|gnb|tower|cell)", 0.20),
        # Stable/ideal conditions
        (r"(?:stable|steady|ideal).{0,30}(?:signal|channel|link|condition)", 0.10),
        # REMOVED: "BLER low" — triggers 73-78% for ALL zones, non-discriminating
        # REMOVED: "DL BLER low" — triggers MORE for zone C (59%) than A (51%)
    ],
    "zone b": [
        # "moderate" RSRP/signal strength is THE key B indicator
        (r"(?:moderate|intermediate)\s+(?:rsrp|signal\s+strength)", 0.15),
        (r"(?:rsrp|signal\s+strength)\s+(?:is\s+)?(?:moderate|intermediate|not.{0,10}(?:strong|weak))", 0.10),
        # RSRP -105 to -114 dB
        (r"-(?:10[5-9]|11[0-4])\.?\d*\s*d[Bb]", 0.30),
        # "not too far" / "not extreme" — must include "not" to avoid matching bare "far"
        (r"not\s+(?:too\s+)?(?:far|extreme|very\s+(?:far|weak))", 0.15),
        (r"intermediate.{0,10}(?:distance|location|range|zone)", 0.10),
        # Moderate distance interpretation
        (r"(?:moderate|intermediate|mid).{0,20}(?:distance|range|location)", 0.10),
        # Explicit "not zone c" or "not extreme" reasoning
        (r"(?:not.{0,15}(?:extreme|very weak|zone c|farthest))", 0.05),
        # Variability/fluctuation in UL metrics
        (r"(?:fluctuat|vari).{0,30}(?:snr|mcs|ul|uplink)", 0.05),
    ],
    "zone c": [
        # RSRP/signal strength specifically "weak" (tight patterns)
        (r"(?:weak|poor|degraded)\s+(?:rsrp|signal\s+strength)", 0.15),
        (r"(?:rsrp|signal\s+strength)\s+(?:is\s+)?(?:weak|very low|poor|degraded)", 0.10),
        # RSRP < -115 dB
        (r"-(?:11[5-9]|1[2-3]\d)\.?\d*\s*d[Bb]", 0.25),
        # Mentioning RSRP below -114 as a specific number
        (r"-(?:114|115|116|117|118|119|12\d)\.?\d*\s*d[Bb]", 0.10),
        # "far/distant" — but NOT "not far" (zone B says "not too far")
        (r"(?<!not\s)(?<!not\s\w\s)(?:far|distant|farthest)\s+(?:from|location|zone|away)", 0.10),
        # Compensation / retransmission
        (r"(?:compensat|retransmiss)", 0.10),
        # Path loss interpretation
        (r"(?:path\s*loss|propagat|attenuat)", 0.10),
        # DL BLER variability — C has more variable DL BLER than B
        (r"(?:dl|downlink).{0,20}(?:bler|error).{0,30}(?:spike|vari|fluctuat)", 0.10),
        # REMOVED: "very/extremely weak/low" — matches "very low BLER" in ALL zones
        # REMOVED: "far/distant/edge" bare — triggers 77% for zone B too
    ],
}

# Patterns that CONTRADICT a zone (penalize if present for GT zone)
_ZONE_NEGATIVE: Dict[str, List[tuple]] = {
    "zone a": [
        # Zone A should NOT say RSRP/signal strength is "weak"
        (r"(?:weak|poor|degraded)\s+(?:rsrp|signal\s+strength)", 0.20),
        (r"(?:rsrp|signal\s+strength)\s+(?:is\s+)?(?:weak|poor|degraded)", 0.15),
        # Zone A should NOT mention RSRP < -114
        (r"-(?:11[5-9]|1[2-3]\d)\.?\d*\s*d[Bb]", 0.25),
        # Zone A should NOT say "moderate signal strength"
        (r"(?:moderate|intermediate)\s+(?:rsrp|signal\s+strength)", 0.10),
        (r"(?:rsrp|signal\s+strength)\s+(?:is\s+)?(?:moderate|mid)", 0.10),
        # REMOVED: bare "far/distant/edge" — triggers 45% for zone A
        # (model says "not too far" or "far from zone C characteristics")
    ],
    "zone b": [
        # Zone B should NOT say RSRP/signal is "weak" — THE failure mode (B→C)
        (r"(?:weak|poor|degraded)\s+(?:rsrp|signal\s+strength)", 0.20),
        (r"(?:rsrp|signal\s+strength)\s+(?:is\s+)?(?:weak|very low|poor|degraded)", 0.15),
        # Zone B should NOT mention RSRP < -115 (zone C territory)
        (r"-(?:11[5-9]|1[2-3]\d)\.?\d*\s*d[Bb]", 0.20),
        # Zone B should NOT say RSRP is "strong"
        (r"(?:strong|very high)\s+(?:rsrp|signal\s+strength)", 0.10),
        (r"(?:rsrp|signal\s+strength)\s+(?:is\s+)?(?:strong|very high)", 0.10),
        # Zone B should NOT mention RSRP > -100 (zone A territory)
        (r"-(?:8\d|9\d)\.?\d*\s*d[Bb]", 0.10),
        # REMOVED: bare "far/distant/edge" — triggers 77% for zone B
        # (model says "not too far" which contains "far")
        # REMOVED: "DL BLER spiking" — only 0.3% trigger, not useful
    ],
    "zone c": [
        # Zone C should NOT say RSRP/signal strength is "strong" or "moderate"
        (r"(?:strong|high|good)\s+(?:rsrp|signal\s+strength)", 0.20),
        (r"(?:rsrp|signal\s+strength)\s+(?:is\s+)?(?:strong|high|good)", 0.15),
        (r"(?:moderate|intermediate)\s+(?:rsrp|signal\s+strength)", 0.10),
        (r"(?:rsrp|signal\s+strength)\s+(?:is\s+)?(?:moderate|mid|intermediate)", 0.10),
        # Zone C should NOT say "close" / "proximity"
        (r"(?:proxim|close|near).{0,15}(?:base|gnb|tower)", 0.15),
        # Zone C should NOT mention RSRP > -105 (zone A territory)
        (r"-(?:8\d|9\d|10[0-4])\.?\d*\s*d[Bb]", 0.15),
    ],
}


def score_zone_reasoning(think_text: str, ground_truth_zone: str) -> float:
    """Score the quality of zone reasoning in the <think> block.

    Uses positive patterns (reward matching GT zone keywords) and negative
    patterns (penalize contradicting keywords).  The negative component is
    critical for preventing zone B/C confusion — it penalizes the model for
    using "weak signal" / "far" language when the GT is zone B.

    Returns 0.0-1.0.
    """
    zone = ground_truth_zone.lower().strip()
    pos_patterns = _ZONE_POSITIVE.get(zone)
    neg_patterns = _ZONE_NEGATIVE.get(zone)
    if not pos_patterns:
        return 0.0

    t = think_text.lower()

    # Positive score: fraction of GT-zone patterns found
    pos_total = sum(w for _, w in pos_patterns)
    pos_earned = sum(w for pat, w in pos_patterns if re.search(pat, t))
    pos_score = pos_earned / pos_total if pos_total > 0 else 0.0

    # Negative penalty: fraction of contradicting patterns found
    neg_total = sum(w for _, w in neg_patterns) if neg_patterns else 1.0
    neg_earned = sum(w for pat, w in neg_patterns if re.search(pat, t)) if neg_patterns else 0.0
    neg_penalty = neg_earned / neg_total if neg_total > 0 else 0.0

    # Combined: positive contribution minus penalty, clamped to [0, 1]
    # Weight: 60% positive, 40% negative penalty
    score = 0.6 * pos_score + 0.4 * (1.0 - neg_penalty)
    return max(0.0, min(1.0, score))


# ── Root cause reasoning quality reward ───────────────────────────────
# 11-class anomaly classification with 4 pairs of identical-KPI anomalies.
# The reward uses:
#   1. Positive patterns: credit for mentioning the GT anomaly's distinctive
#      keywords (anomaly name, temporal patterns, mechanism).
#   2. Negative patterns: penalty for mentioning a CONFUSING anomaly's
#      distinctive keywords.
#
# Key confusing pairs (Jaccard similarity of affected KPIs):
#   - Co-Channel Interference (Mild) ↔ Faulty RF Filters (1.00)
#     Discriminator: CCI is step-change, RF Filters is linear/progressive
#   - Buffer Overflow ↔ High Network Congestion Gradual (1.00)
#     Discriminator: Buffer is buffer-centric, Congestion is network-centric
#   - Congestion Gradual ↔ Congestion Sudden (1.00)
#     Discriminator: gradual buildup vs sudden spike
#   - CCI Mild ↔ CCI Severe (0.82)
#     Discriminator: magnitude (mild/small vs severe/significant)

_RC_POSITIVE: Dict[str, List[tuple]] = {
    "antenna failure": [
        (r"\bantenna\b", 0.25),
        (r"(?:hardware|component|physical)\s+(?:failure|fault|malfunction|damage)", 0.15),
        (r"rsrp\s+(?:drop|fell|decreas|declin)", 0.15),
        (r"(?:both|ul\s+and\s+dl|uplink\s+and\s+downlink)\s+(?:bler|error)", 0.10),
        (r"(?:mcs|modulation)\s+(?:decreas|drop|fell)", 0.10),
        (r"(?:buffer|backlog)\s+(?:grew|increas|exponential)", 0.10),
        (r"(?:surge|exponential).{0,30}(?:bler|error\s+rate)", 0.10),
        (r"(?:packet\s+count|packets)\s+(?:increas|rose)", 0.05),
    ],
    "co-channel interference (mild)": [
        (r"co.?channel\s+interference", 0.20),
        (r"\bmild\b|\bslight\b|\bmoderate\b|\bminor\b", 0.15),
        (r"(?:small|slight|moderate)\s+(?:decrease|drop|degradation)", 0.15),
        (r"(?:3.?8|1.?3)\s*d[Bb]", 0.10),  # characteristic magnitude
        (r"interference.{0,30}(?:uplink|ul)", 0.10),
        (r"(?:collision|hidden\s+node|adjacent\s+cell)", 0.10),
        (r"(?:prb|resource).{0,20}(?:utilization|allocation).{0,20}increas", 0.10),
        (r"(?:step\s+change|constant\s+offset|flat\s+degradation)", 0.10),
    ],
    "co-channel interference (severe)": [
        (r"co.?channel\s+interference", 0.15),
        (r"\bsevere\b|\bsignificant\b|\bmajor\b|\bdrastic\b|\bsubstantial\b", 0.20),
        (r"(?:large|significant|major)\s+(?:decrease|drop|degradation)", 0.15),
        (r"(?:8.?15|50.?200)\s*(?:d[Bb]|%)", 0.10),  # characteristic magnitude
        (r"(?:packet\s+count|packets).{0,20}(?:drop|decreas|declin)", 0.10),
        (r"interference.{0,30}(?:both|uplink\s+and\s+downlink)", 0.10),
        (r"(?:prb|resource).{0,20}(?:utilization|allocation).{0,20}(?:30.?80|high)", 0.10),
        (r"(?:throughput|bytes).{0,20}(?:30.?80|significant).{0,10}(?:drop|decreas|decline)", 0.10),
    ],
    "faulty rf filters (temporal)": [
        (r"(?:rf|radio\s+frequency)\s+filter", 0.25),
        (r"\blinear\b.{0,30}(?:decrease|drop|decline|decay)", 0.15),
        (r"\bexponential\b.{0,30}(?:increase|growth|rise)", 0.15),
        (r"\blogarithm\b.{0,30}(?:decay|decrease|decline)", 0.10),
        (r"(?:progressive|temporal|time.?varying|gradual\s+degradation)", 0.15),
        (r"(?:filter|passband|frequency\s+response).{0,20}(?:degrad|fault|malfunc)", 0.10),
        (r"(?:hardware|component).{0,20}(?:degradation|aging|failure)", 0.10),
    ],
    "doppler shift (severe)": [
        (r"\bdoppler\b", 0.30),
        (r"(?:periodic|oscillat|cyclic)\s+(?:fluctuat|variation|change)", 0.15),
        (r"(?:2.?3)\s*(?:Hz|hertz)", 0.10),  # characteristic frequency
        (r"(?:movement|velocity|speed|motion|mobile)", 0.15),
        (r"(?:frequency\s+shift|wavelength\s+change)", 0.10),
        (r"(?:multiplicative|factor\s+0\.\d)", 0.10),
        (r"(?:mcs|modulation)\s+(?:decreas|drop|oscillat)", 0.10),
    ],
    "faulty handover algorithm (too frequent)": [
        (r"\bhandover\b|\bhandoff\b", 0.30),
        (r"(?:too\s+frequent|excessive|repeated)\s+(?:handover|transition|switch)", 0.15),
        (r"(?:periodic|oscillat).{0,30}(?:rsrp|signal|snr)", 0.15),
        (r"(?:0\.1.?0\.3)\s*(?:Hz|hertz)", 0.05),  # characteristic frequency
        (r"(?:ping.?pong|back.?and.?forth|repeated\s+switch)", 0.10),
        (r"(?:buffer|backlog).{0,20}(?:increas|grew|spike)", 0.10),
        (r"(?:reconfigur|re.?attach|context\s+switch)", 0.10),
        (r"(?:bursty|intermittent).{0,20}(?:resource|prb|allocation)", 0.05),
    ],
    "buffer overflow (gradual buildup)": [
        (r"\bbuffer\b.{0,15}(?:overflow|saturat|exhaust|full)", 0.25),
        (r"(?:buffer|backlog|queue).{0,20}(?:exponential|gradual|progressive)", 0.15),
        (r"(?:gradual|progressive|over\s+time).{0,20}(?:buildup|growth|accumulat)", 0.15),
        (r"(?:memory|queue|backlog).{0,20}(?:full|exhaust|capacity)", 0.10),
        (r"(?:logistic|s.?curve|sigmoid).{0,20}(?:prb|utilization)", 0.10),
        (r"(?:logarithm|decay).{0,20}(?:bytes|throughput)", 0.10),
        (r"(?:linear\s+increase).{0,20}(?:packet|count)", 0.10),
        (r"(?:buffer.?centric|single\s+user|local)", 0.05),
    ],
    "high network congestion (gradual buildup)": [
        (r"\bcongestion\b", 0.20),
        (r"(?:gradual|progressive|over\s+time).{0,20}(?:buildup|growth|increase)", 0.15),
        (r"(?:network|cell).{0,15}(?:congestion|overload|capacity)", 0.15),
        (r"(?:prb|resource).{0,20}(?:utilization|allocation).{0,20}(?:logistic|growth|increase)", 0.10),
        (r"(?:exponential).{0,20}(?:buffer|backlog|error)", 0.10),
        (r"(?:shared\s+resource|contention|competing\s+user)", 0.10),
        (r"(?:capacity|limit).{0,20}(?:reach|approach|near)", 0.10),
        (r"(?:network.?wide|cell.?level|multiple\s+user)", 0.10),
    ],
    "high network congestion (sudden spike)": [
        (r"\bcongestion\b", 0.15),
        (r"(?:sudden|abrupt|sharp|instantaneous).{0,20}(?:spike|increase|onset|jump)", 0.25),
        (r"(?:network|cell).{0,15}(?:congestion|overload)", 0.10),
        (r"\bsurge\b|\bspike\b", 0.15),
        (r"(?:step\s+increase|rapid\s+onset|flash\s+crowd)", 0.10),
        (r"(?:100.?400|30.?80).{0,10}%?\s*(?:increase|spike)", 0.10),
        (r"(?:prb|resource).{0,20}(?:utilization|allocation).{0,20}(?:spike|jump|surge)", 0.10),
        (r"(?:transient|short.?lived|brief)", 0.05),
    ],
    "resource allocation bugs": [
        (r"(?:resource\s+allocation|scheduling).{0,15}(?:bug|error|fault|malfunction)", 0.25),
        (r"(?:multiplicative|periodic)\s+(?:oscillat|fluctuat)", 0.20),
        (r"(?:0\.3.?1\.0)\s*(?:Hz|hertz)", 0.05),  # characteristic frequency
        (r"(?:prb|resource).{0,20}(?:oscillat|fluctuat|erratic|irregular)", 0.15),
        (r"(?:bytes|throughput).{0,20}(?:oscillat|fluctuat|erratic)", 0.10),
        (r"(?:software|algorithm|implementation).{0,20}(?:bug|error|defect)", 0.10),
        (r"(?:scheduling|dispatcher|controller).{0,20}(?:fault|error|malfunction)", 0.10),
        (r"(?:systematic|deterministic|reproducible)", 0.05),
    ],
    "jamming": [
        (r"\bjamming\b|\bjammer\b", 0.30),
        (r"(?:deliberate|intentional|malicious|external).{0,20}(?:interference|attack|signal)", 0.15),
        (r"(?:artificial|non.?natural|man.?made).{0,20}(?:interference|noise|signal)", 0.10),
        (r"(?:dl|downlink).{0,15}(?:bler|error).{0,20}(?:increas|spike|surge)", 0.10),
        (r"(?:ul|uplink).{0,15}mcs.{0,20}(?:decreas|drop)", 0.10),
        (r"(?:tx|rx|bytes|throughput).{0,20}(?:drop|collapse|zero)", 0.10),
        (r"(?:broadband|wideband|full.?band).{0,20}(?:noise|interference)", 0.10),
        (r"(?:security|threat|attack|adversar)", 0.05),
    ],
}

# Negative patterns: penalize when the model uses keywords from a CONFUSING
# anomaly.  Focus on the 4 confusing pairs with identical/near-identical KPIs.
_RC_NEGATIVE: Dict[str, List[tuple]] = {
    # CCI Mild should NOT use RF filter language
    "co-channel interference (mild)": [
        (r"(?:rf|radio\s+frequency)\s+filter", 0.25),
        (r"\blinear\b.{0,30}(?:decrease|decay|drop)", 0.15),
        (r"\bexponential\b.{0,30}(?:increase|growth)", 0.15),
        (r"\blogarithm", 0.10),
        (r"\bsevere\b|\bdrastic\b|\bmajor\b", 0.15),  # Not severe CCI
    ],
    # RF Filters should NOT use CCI language
    "faulty rf filters (temporal)": [
        (r"co.?channel\s+interference", 0.25),
        (r"\bcollision\b|\bhidden\s+node\b", 0.15),
        (r"\bmild\b.{0,20}(?:interference|degradation)", 0.15),
        (r"(?:step\s+change|constant\s+offset)", 0.10),
    ],
    # Buffer Overflow should NOT use congestion language
    "buffer overflow (gradual buildup)": [
        (r"(?:network|cell).{0,15}(?:congestion|overload)", 0.20),
        (r"(?:shared\s+resource|contention|competing)", 0.15),
        (r"(?:network.?wide|cell.?level|multiple\s+user)", 0.15),
        (r"\bsudden\b|\babrupt\b|\bspike\b", 0.15),  # Not sudden
    ],
    # Congestion Gradual should NOT use buffer-overflow or sudden language
    "high network congestion (gradual buildup)": [
        (r"\bbuffer\b.{0,15}(?:overflow|exhaust|full)", 0.20),
        (r"(?:single\s+user|buffer.?centric|memory)", 0.15),
        (r"\bsudden\b|\babrupt\b|\bspike\b|\binstantaneous\b", 0.20),
    ],
    # Congestion Sudden should NOT use gradual language
    "high network congestion (sudden spike)": [
        (r"\bgradual\b|\bprogressive\b|\bslowly\b|\bover\s+time\b", 0.25),
        (r"\bbuffer\b.{0,15}(?:overflow|exhaust)", 0.15),
        (r"(?:exponential\s+growth|logistic)", 0.10),
    ],
    # CCI Severe should NOT use mild language
    "co-channel interference (severe)": [
        (r"\bmild\b|\bslight\b|\bminor\b", 0.20),
        (r"(?:small|minimal|negligible)\s+(?:decrease|impact)", 0.15),
        (r"(?:rf|radio\s+frequency)\s+filter", 0.15),
    ],
    # Antenna Failure should NOT use handover language
    "antenna failure": [
        (r"\bhandover\b|\bhandoff\b", 0.20),
        (r"(?:ping.?pong|too\s+frequent)", 0.15),
        (r"co.?channel\s+interference", 0.15),
    ],
    # Handover should NOT use antenna failure language
    "faulty handover algorithm (too frequent)": [
        (r"\bantenna\b.{0,15}(?:failure|fault|damage)", 0.20),
        (r"(?:hardware|component).{0,15}(?:failure|fault)", 0.15),
        (r"\bdoppler\b", 0.15),
    ],
    # Doppler should NOT use handover language
    "doppler shift (severe)": [
        (r"\bhandover\b|\bhandoff\b", 0.15),
        (r"(?:too\s+frequent|excessive)\s+(?:handover|switch)", 0.15),
        (r"\bcongestion\b", 0.10),
    ],
    # Resource allocation should NOT use congestion/buffer language
    "resource allocation bugs": [
        (r"\bcongestion\b", 0.15),
        (r"\bbuffer\b.{0,15}(?:overflow|exhaust)", 0.15),
        (r"co.?channel\s+interference", 0.10),
    ],
    # Jamming should NOT use specific other anomaly names
    "jamming": [
        (r"\bhandover\b", 0.15),
        (r"\bantenna\b.{0,15}failure", 0.15),
        (r"\bcongestion\b", 0.10),
        (r"(?:rf|radio\s+frequency)\s+filter", 0.10),
    ],
}


def score_root_cause_reasoning(think_text: str, ground_truth_anomaly: str) -> float:
    """Score root cause reasoning quality in the <think> block.

    Uses positive patterns (reward matching GT anomaly keywords) and negative
    patterns (penalize keywords from confusing anomalies).

    Returns 0.0-1.0.
    """
    anomaly = ground_truth_anomaly.lower().strip()

    # Normalize GT to canonical form
    canonical = None
    for key in _RC_POSITIVE:
        if key in anomaly or anomaly in key:
            canonical = key
            break
    if canonical is None:
        # Try partial match
        for key in _RC_POSITIVE:
            if any(w in anomaly for w in key.split()):
                canonical = key
                break
    if canonical is None:
        return 0.0

    pos_patterns = _RC_POSITIVE.get(canonical, [])
    neg_patterns = _RC_NEGATIVE.get(canonical, [])

    t = think_text.lower()

    # Positive score: fraction of GT-anomaly patterns found
    pos_total = sum(w for _, w in pos_patterns) if pos_patterns else 1.0
    pos_earned = sum(w for pat, w in pos_patterns if re.search(pat, t))
    pos_score = pos_earned / pos_total if pos_total > 0 else 0.0

    # Negative penalty: fraction of contradicting patterns found
    neg_total = sum(w for _, w in neg_patterns) if neg_patterns else 1.0
    neg_earned = sum(w for pat, w in neg_patterns if re.search(pat, t)) if neg_patterns else 0.0
    neg_penalty = neg_earned / neg_total if neg_total > 0 else 0.0

    # Combined: 60% positive, 40% negative penalty
    score = 0.6 * pos_score + 0.4 * (1.0 - neg_penalty)
    return max(0.0, min(1.0, score))


# ── Anomaly detection reasoning quality reward ──────────────────────
# Binary: does an anomaly exist?
#
# When GT is "yes", we know the anomaly_type from extra_info, so we can
# reward the model for mentioning the correct affected KPIs and symptoms.
# When GT is "no", reward stability / normal-range language.
#
# Uses anomaly_type_2_affected_kpis and anomaly_type_2_symptom from
# qa_templates.py for per-anomaly KPI checking.

# Import symptom descriptions
from teleprism.dataset.qa_templates import (
    reverse_anomalies_type_2_id,
)

# Symptom keyword patterns per anomaly type (extracted from anomaly_type_2_symptom)
_AD_SYMPTOM_PATTERNS: Dict[int, List[tuple]] = {
    1: [  # Antenna Failure
        (r"(?:rsrp|signal\s+(?:strength|power))\s+(?:drop|fell|decreas|declin)", 0.20),
        (r"(?:snr|signal.to.noise)\s+(?:fell|drop|decreas)", 0.15),
        (r"(?:bler|error\s+rate)\s+(?:surge|spike|exponential|increas)", 0.20),
        (r"(?:mcs|modulation)\s+(?:decreas|drop|fell)", 0.15),
        (r"(?:buffer|backlog)\s+(?:grew|growth|increas|exponential)", 0.15),
        (r"(?:bytes|throughput)\s+(?:declin|decay|logarithm|decreas)", 0.15),
    ],
    2: [  # CCI Mild
        (r"(?:rsrp|signal)\s+(?:decreas|drop|reduc).{0,20}(?:3.?8|small|slight|mild)", 0.20),
        (r"(?:bler|error\s+rate)\s+(?:increas|higher|elevat).{0,20}(?:10.?40|mild|slight)", 0.20),
        (r"(?:prb|resource)\s+(?:utilization|allocation)\s+(?:increas|higher|5.?20)", 0.15),
        (r"(?:bytes|throughput)\s+(?:reduc|decreas|lower).{0,20}(?:5.?15|slight)", 0.15),
        (r"(?:step.?change|constant\s+offset|flat)", 0.15),
        (r"(?:interference|collision|co.?channel)", 0.15),
    ],
    3: [  # CCI Severe
        (r"(?:rsrp|signal)\s+(?:signific|large|major)\s+(?:decreas|drop)", 0.20),
        (r"(?:bler|error\s+rate)\s+(?:50.?200|signific|major)\s*%?\s*(?:increas)?", 0.15),
        (r"(?:bytes|throughput)\s+(?:30.?80|signific|major)\s*%?\s*(?:drop|decreas|declin)", 0.15),
        (r"(?:prb|resource)\s+(?:utilization|allocation).{0,20}(?:30.?80|signific|increas)", 0.15),
        (r"(?:packet\s+count|packets)\s+(?:40.?70|drop|decreas)", 0.15),
        (r"(?:severe|significant|major)\s+(?:interference|degradation)", 0.20),
    ],
    4: [  # Faulty RF Filters
        (r"(?:linear|progressive)\s+(?:decreas|drop|decay)", 0.20),
        (r"(?:exponential)\s+(?:increas|growth|rise).{0,30}(?:bler|error)", 0.20),
        (r"(?:logarithm)\s+(?:decay|decreas).{0,30}(?:bytes|throughput)", 0.15),
        (r"(?:linear\s+increas).{0,20}(?:prb|allocation)", 0.15),
        (r"(?:temporal|progressive|time.?varying|gradual)\s+(?:degradation|decay|decline)", 0.15),
        (r"(?:filter|rf\s+filter|passband)", 0.15),
    ],
    5: [  # Doppler Shift
        (r"(?:periodic|oscillat|cyclic)\s+(?:fluctuat|variation)", 0.20),
        (r"(?:2.?3)\s*(?:Hz|hertz)", 0.10),
        (r"(?:rsrp|signal).{0,20}(?:3.?8)\s*d[Bb]", 0.15),
        (r"(?:mcs|modulation)\s+(?:decreas|drop)", 0.15),
        (r"(?:multiplicat|factor\s+0\.\d).{0,20}(?:bytes|prb|oscillat)", 0.15),
        (r"(?:doppler|frequency\s+shift|velocity|movement)", 0.25),
    ],
    6: [  # Faulty Handover
        (r"(?:periodic|oscillat)\s+(?:rsrp|signal|snr)", 0.20),
        (r"(?:0\.1.?0\.3)\s*(?:Hz|hertz)", 0.10),
        (r"(?:bler|error\s+rate)\s+(?:30.?150|increas)", 0.15),
        (r"(?:buffer|backlog)\s+(?:50.?200|increas|grew)", 0.15),
        (r"(?:bytes|throughput)\s+(?:10.?50|decreas|reduc)", 0.10),
        (r"(?:handover|handoff|ping.?pong|frequent\s+switch)", 0.30),
    ],
    7: [  # Buffer Overflow
        (r"(?:exponential)\s+(?:growth|increas).{0,30}(?:buffer|backlog|bler|error)", 0.20),
        (r"(?:logarithm|decay).{0,20}(?:bytes|throughput)", 0.15),
        (r"(?:linear\s+increas).{0,20}(?:packet|count)", 0.15),
        (r"(?:logistic)\s+(?:growth|curve).{0,20}(?:prb|utilization)", 0.15),
        (r"(?:buffer)\s+(?:overflow|saturat|exhaust|full|gradual)", 0.20),
        (r"(?:gradual|progressive)\s+(?:buildup|accumulation|growth)", 0.15),
    ],
    8: [  # Resource Allocation Bugs
        (r"(?:multiplicat|periodic)\s+(?:oscillat|fluctuat)", 0.20),
        (r"(?:0\.3.?1\.0)\s*(?:Hz|hertz)", 0.10),
        (r"(?:prb|resource).{0,20}(?:oscillat|fluctuat|erratic)", 0.20),
        (r"(?:bytes|throughput)\s+(?:oscillat|fluctuat)", 0.15),
        (r"(?:bler|error)\s+(?:increas|20.?100)", 0.15),
        (r"(?:resource\s+allocation|scheduling)\s+(?:bug|error|fault)", 0.20),
    ],
    9: [  # Congestion Gradual
        (r"(?:exponential)\s+(?:growth|increas).{0,30}(?:buffer|backlog|bler|error|packet)", 0.20),
        (r"(?:logistic)\s+(?:growth|curve).{0,20}(?:prb|utilization|allocation)", 0.15),
        (r"(?:logarithm|decay).{0,20}(?:bytes|throughput)", 0.15),
        (r"(?:gradual|progressive)\s+(?:buildup|growth|congestion)", 0.20),
        (r"(?:network|cell)\s+(?:congestion|overload)", 0.15),
        (r"(?:shared\s+resource|contention|multiple\s+user)", 0.15),
    ],
    10: [  # Congestion Sudden
        (r"(?:sudden|abrupt|sharp|instantaneous)\s+(?:spike|increase|onset|jump)", 0.25),
        (r"(?:buffer|backlog)\s+(?:100.?400|signific|massive)\s*%?\s*(?:increas)?", 0.15),
        (r"(?:prb|utilization)\s+(?:30.?80|signific)\s*%?\s*(?:increas)?", 0.15),
        (r"(?:bler|error)\s+(?:50.?150|signific)\s*%?\s*(?:increas)?", 0.15),
        (r"(?:bytes|throughput)\s+(?:20.?50|signific)\s*%?\s*(?:drop|decreas)", 0.15),
        (r"(?:step\s+increase|flash|surge|spike)", 0.15),
    ],
    11: [  # Jamming
        (r"(?:dl|downlink)\s+(?:bler|error)\s+(?:increas|spike|surge)", 0.20),
        (r"(?:ul|uplink)\s+(?:bler|error)\s+(?:increas|spike|surge)", 0.15),
        (r"(?:ul|uplink)\s+(?:mcs|modulation)\s+(?:decreas|drop)", 0.15),
        (r"(?:tx|rx|bytes|throughput)\s+(?:drop|collapse|decreas)", 0.15),
        (r"(?:jamming|jammer|deliberate|malicious|external\s+interference)", 0.25),
        (r"(?:broadband|wideband|full.?band)\s+(?:noise|interference)", 0.10),
    ],
}

# "No anomaly" patterns: stability language
_AD_NO_ANOMALY_POSITIVE = [
    (r"(?:rsrp|signal)\s+(?:stable|steady|consistent|constant)", 0.15),
    (r"(?:bler|error\s+rate)\s+(?:remains?\s+)?(?:low|zero|0\.0|nominal|minimal)", 0.15),
    (r"(?:within|inside)\s+(?:expected|normal|typical|healthy|operational)\s+(?:range|limit|threshold)", 0.15),
    (r"(?:no|without)\s+(?:significant\s+)?(?:degradation|deviation|anomal|spike|abnormal)", 0.15),
    (r"(?:nominal|normal|stable|healthy)\s+(?:performance|condition|operation|behavior)", 0.10),
    (r"(?:manageable|stable|low)\s+(?:buffer|backlog|queue)", 0.10),
    (r"(?:consistent|uniform|steady)\s+(?:traffic|throughput|pattern|allocation)", 0.10),
    (r"(?:no\s+)?(?:abrupt|sudden|rapid)\s+(?:change|variation)", 0.10),
]

_AD_NO_ANOMALY_NEGATIVE = [
    (r"(?:sharp|abrupt|sudden)\s+(?:drop|spike|increase|degradation)", 0.25),
    (r"(?:abnormal|anomal|irregular)\s+(?:behavior|pattern|value)", 0.20),
    (r"(?:exponential|surge|collapse|overflow|saturat)", 0.20),
    (r"(?:degradation|instability|failure)", 0.15),
]


def score_anomaly_detection_reasoning(think_text: str, ground_truth: str,
                                      anomaly_type: int = 0) -> float:
    """Score anomaly detection reasoning.

    When GT="yes" and anomaly_type is known (1-11), checks if the model
    identifies the correct affected KPIs and symptoms for that anomaly type.
    When GT="no", checks for stability language.
    Returns 0.0-1.0.
    """
    gt = ground_truth.lower().strip()
    is_anomaly = "yes" in gt or "anomaly detected" in gt
    t = think_text.lower()

    if is_anomaly and anomaly_type in _AD_SYMPTOM_PATTERNS:
        # Use anomaly-specific symptom patterns
        pos_patterns = _AD_SYMPTOM_PATTERNS[anomaly_type]
        # Also check affected KPI mentions
        anomaly_name = reverse_anomalies_type_2_id.get(anomaly_type, "")
        affected_kpis = anomaly_type_2_affected_kpis.get(anomaly_name, [])
        kpi_mentioned = count_kpi_mentions(t, affected_kpis)
        kpi_fraction = min(1.0, kpi_mentioned / max(1, len(affected_kpis) * 0.4))

        pos_total = sum(w for _, w in pos_patterns) if pos_patterns else 1.0
        pos_earned = sum(w for pat, w in pos_patterns if re.search(pat, t))
        pos_score = pos_earned / pos_total if pos_total > 0 else 0.0

        # 50% symptom pattern match, 30% KPI mention, 20% baseline (no stability claim)
        stability_claim = 1.0 if re.search(
            r"(?:no|without)\s+(?:significant\s+)?(?:degradation|deviation|anomal)", t
        ) else 0.0
        score = 0.50 * pos_score + 0.30 * kpi_fraction + 0.20 * (1.0 - stability_claim)

    elif is_anomaly:
        # Anomaly present but unknown type — generic deviation language
        generic_patterns = [
            (r"(?:bler|error\s+rate)\s+(?:rises?|elevated|spikes?|increas|above)", 0.15),
            (r"(?:buffer|backlog)\s+(?:spikes?|grows?|increas|elevated)", 0.15),
            (r"(?:throughput|bytes)\s+(?:drops?|decreas|declin)", 0.15),
            (r"(?:sharp|abrupt|sudden|rapid)\s+(?:change|drop|spike|increase|degradation)", 0.20),
            (r"(?:degradation|instability|abnormal|anomal)", 0.15),
            (r"(?:above|exceed|beyond)\s+(?:threshold|expected|typical|normal)", 0.10),
            (r"(?:correlat|coincid|aligns?\s+with)", 0.10),
        ]
        pos_total = sum(w for _, w in generic_patterns)
        pos_earned = sum(w for pat, w in generic_patterns if re.search(pat, t))
        score = pos_earned / pos_total if pos_total > 0 else 0.0

    else:
        # No anomaly
        pos_total = sum(w for _, w in _AD_NO_ANOMALY_POSITIVE)
        pos_earned = sum(w for pat, w in _AD_NO_ANOMALY_POSITIVE if re.search(pat, t))
        pos_score = pos_earned / pos_total if pos_total > 0 else 0.0

        neg_total = sum(w for _, w in _AD_NO_ANOMALY_NEGATIVE)
        neg_earned = sum(w for pat, w in _AD_NO_ANOMALY_NEGATIVE if re.search(pat, t))
        neg_penalty = neg_earned / neg_total if neg_total > 0 else 0.0

        score = 0.6 * pos_score + 0.4 * (1.0 - neg_penalty)

    return max(0.0, min(1.0, score))


# ── Motion reasoning quality reward ─────────────────────────────────
# Binary: stationary vs mobile.
#
# From cold-start CoT (1854 traces):
#   Stationary: "RSRP remained steady/stable", "fixed/constant MCS",
#               "no rapid fading", "tightly clustered SNR", "no Doppler"
#   Mobile:     "RSRP decline/recovery/fluctuation", "abrupt MCS transitions",
#               "rapid channel quality shifts", "Doppler", "erratic PRB"

_MOTION_POSITIVE: Dict[str, List[tuple]] = {
    "stationary": [
        (r"rsrp\s+(?:remain|stay|held|is)?\s*(?:steady|stable|constant|consistent|fixed)", 0.20),
        (r"(?:stable|steady|consistent|constant)\s+(?:rsrp|signal\s+strength)", 0.15),
        (r"(?:no|without|minimal)\s+(?:rapid\s+)?(?:fading|doppler|handover)", 0.15),
        (r"(?:no|minimal)\s+(?:environmental|channel)\s+(?:change|variation)", 0.10),
        (r"(?:fixed|constant|stable)\s+(?:mcs|modulation)", 0.10),
        (r"(?:tightly|narrowly)\s+(?:clustered|distributed|bounded)", 0.10),
        (r"(?:static|stationary)\s+(?:channel|conditions|environment|position)", 0.10),
        (r"(?:no|minimal)\s+(?:significant\s+)?(?:mobility|movement|motion)", 0.10),
    ],
    "mobile": [
        (r"rsrp\s+(?:decline|drop|fluctuat|vari|recover|shift|oscillat)", 0.15),
        (r"(?:signal|rsrp)\s+(?:degradation|variation|fluctuation|instability)", 0.15),
        (r"(?:rapid|abrupt|sharp)\s+(?:channel|mcs|signal|snr)\s+(?:change|transition|shift|quality)", 0.15),
        (r"(?:erratic|irregular|bursty)\s+(?:prb|resource|allocation|fluctuation)", 0.10),
        (r"\bdoppler\b", 0.15),
        (r"(?:movement|velocity|speed|motion|moving|mobile)", 0.10),
        (r"(?:handover|cell\s+transition|reselection)", 0.10),
        (r"(?:unstable|intermittent)\s+(?:link|signal|channel|condition)", 0.10),
    ],
}

_MOTION_NEGATIVE: Dict[str, List[tuple]] = {
    "stationary": [
        (r"rsrp\s+(?:decline|drop|fluctuat|oscillat)", 0.20),
        (r"(?:rapid|abrupt)\s+(?:change|transition|shift)", 0.15),
        (r"\bdoppler\b", 0.15),
        (r"(?:movement|mobile|in\s+motion)", 0.15),
    ],
    "mobile": [
        (r"rsrp\s+(?:remain|stay)?\s*(?:steady|stable|constant)", 0.20),
        (r"(?:no|without|minimal)\s+(?:mobility|movement|motion|fading)", 0.20),
        (r"(?:static|stationary)\s+(?:channel|conditions|position)", 0.15),
    ],
}


def score_motion_reasoning(think_text: str, ground_truth: str) -> float:
    """Score motion reasoning. Returns 0.0-1.0."""
    gt = ground_truth.lower().strip()
    gt_class = "mobile" if ("mobile" in gt or "moving" in gt or "motion" in gt or gt == "yes") else "stationary"

    pos_patterns = _MOTION_POSITIVE.get(gt_class, [])
    neg_patterns = _MOTION_NEGATIVE.get(gt_class, [])
    t = think_text.lower()

    pos_total = sum(w for _, w in pos_patterns) if pos_patterns else 1.0
    pos_earned = sum(w for pat, w in pos_patterns if re.search(pat, t))
    pos_score = pos_earned / pos_total if pos_total > 0 else 0.0

    neg_total = sum(w for _, w in neg_patterns) if neg_patterns else 1.0
    neg_earned = sum(w for pat, w in neg_patterns if re.search(pat, t)) if neg_patterns else 0.0
    neg_penalty = neg_earned / neg_total if neg_total > 0 else 0.0

    score = 0.6 * pos_score + 0.4 * (1.0 - neg_penalty)
    return max(0.0, min(1.0, score))


# ── Congestion reasoning quality reward ─────────────────────────────
# Binary: congested vs not.
#
# From cold-start CoT (5562 traces):
#   Congestion: "PRB utilization above 35%", "elevated resource demand",
#               "DL_NumberOfPackets spike", "resource contention",
#               "buffer overflow/growth", "throughput drops under load"
#   No Cong:    "low BLER", "no saturation", "resources not overcommitted",
#               "manageable buffer", "within expected ranges"

_CONG_POSITIVE: Dict[str, List[tuple]] = {
    "congestion": [
        (r"(?:prb|resource)\s+(?:utilization|allocation).{0,20}(?:high|elevated|above|peak|surge|saturat)", 0.20),
        (r"(?:utilization|occupancy)\s+(?:above|exceed|peak).{0,15}(?:\d+%|30|40|50)", 0.10),
        (r"(?:resource|capacity)\s+(?:contention|demand|pressure|scarcity|overcommit)", 0.15),
        (r"(?:elevated|high|increased)\s+(?:resource\s+)?demand", 0.10),
        (r"(?:buffer|backlog)\s+(?:growth|increas|overflow|elevated)", 0.10),
        (r"(?:throughput|bytes)\s+(?:drop|decreas|declin|reduc).{0,20}(?:load|demand|congest)", 0.10),
        (r"(?:packet|traffic)\s+(?:loss|drop|spike|surge|demand)", 0.10),
        (r"(?:asymmetr|imbalanc).{0,20}(?:load|traffic|resource|utilization)", 0.05),
        (r"(?:network|cell)\s+(?:overload|saturat|congest|capacity\s+limit)", 0.10),
    ],
    "no congestion": [
        (r"(?:prb|resource)\s+(?:utilization|allocation).{0,20}(?:low|normal|moderate|within|manageable)", 0.15),
        (r"(?:no|without)\s+(?:resource\s+)?(?:saturat|contention|overload|congest)", 0.20),
        (r"(?:bler|error\s+rate)\s+(?:remains?\s+)?(?:low|zero|minimal|0\.0)", 0.15),
        (r"(?:within|inside)\s+(?:expected|normal|manageable|operational)\s+(?:range|limit)", 0.15),
        (r"(?:manageable|stable|low|normal)\s+(?:buffer|backlog|queue)", 0.10),
        (r"(?:resource|capacity)\s+(?:available|sufficient|adequate|not\s+overcommit)", 0.10),
        (r"(?:stable|consistent|nominal)\s+(?:throughput|performance|allocation)", 0.10),
        (r"(?:no|without)\s+(?:overflow|backlog\s+growth|queue\s+buildup)", 0.05),
    ],
}

_CONG_NEGATIVE: Dict[str, List[tuple]] = {
    "congestion": [
        (r"(?:no|without)\s+(?:resource\s+)?(?:saturat|contention|overload|congest)", 0.20),
        (r"(?:stable|consistent|nominal)\s+(?:throughout|overall|across)", 0.15),
        (r"(?:within|inside)\s+(?:expected|normal|manageable)\s+(?:range|limit)", 0.15),
    ],
    "no congestion": [
        (r"(?:resource|capacity)\s+(?:contention|overcommit|saturat|exhaust)", 0.20),
        (r"(?:network|cell)\s+(?:overload|congest)", 0.15),
        (r"(?:prb|resource)\s+(?:utilization|allocation).{0,20}(?:high|peak|surge|saturat)", 0.15),
    ],
}


def score_cong_reasoning(think_text: str, ground_truth: str) -> float:
    """Score congestion reasoning. Returns 0.0-1.0."""
    gt = ground_truth.lower().strip()
    gt_class = "no congestion" if ("no" in gt and "congestion" in gt) or gt.startswith("no") else "congestion"

    pos_patterns = _CONG_POSITIVE.get(gt_class, [])
    neg_patterns = _CONG_NEGATIVE.get(gt_class, [])
    t = think_text.lower()

    pos_total = sum(w for _, w in pos_patterns) if pos_patterns else 1.0
    pos_earned = sum(w for pat, w in pos_patterns if re.search(pat, t))
    pos_score = pos_earned / pos_total if pos_total > 0 else 0.0

    neg_total = sum(w for _, w in neg_patterns) if neg_patterns else 1.0
    neg_earned = sum(w for pat, w in neg_patterns if re.search(pat, t)) if neg_patterns else 0.0
    neg_penalty = neg_earned / neg_total if neg_total > 0 else 0.0

    score = 0.6 * pos_score + 0.4 * (1.0 - neg_penalty)
    return max(0.0, min(1.0, score))


# ── Activity reasoning quality reward ───────────────────────────────
# 3-class: youtube / twitch / file download.
#
# From cold-start CoT (5000 traces):
#   File:    "TX_Bytes >> RX_Bytes", "asymmetric traffic favoring upload",
#            "TCP at 100%", "sustained high throughput", "upload-dominated"
#   Twitch:  "transient high-volume bursts", "PRBs_DL spikes", "bursty/irregular",
#            "interactive", "real-time"
#   YouTube: "adaptive bitrate", "DL PRB utilization spiked",
#            "sustained streaming", "downlink resource allocation"

_ACTIVITY_POSITIVE: Dict[str, List[tuple]] = {
    "file": [
        (r"(?:tx|transmit|upload).{0,20}(?:exceed|>>|far\s+exceed|domin|higher\s+than)", 0.20),
        (r"(?:asymmetr|imbalanc).{0,20}(?:upload|uplink|transmis)", 0.15),
        (r"(?:upload|uplink).{0,15}(?:dominat|heavy|traffic|burst)", 0.15),
        (r"(?:sustained|continuous|steady|large)\s+(?:throughput|transfer|upload|download)", 0.15),
        (r"(?:file|data)\s+(?:transfer|download|upload)", 0.10),
        (r"(?:tcp|protocol)\s+(?:at\s+)?100%", 0.10),
        (r"(?:bulk|large)\s+(?:data|file|transfer)", 0.10),
        (r"(?:tx_bytes|transmitted\s+bytes).{0,20}(?:high|large|mean\s+\d{5,})", 0.05),
    ],
    "twitch": [
        (r"(?:bursty|transient|irregular|intermittent)\s+(?:traffic|burst|pattern|spike)", 0.20),
        (r"(?:prb|resource).{0,20}(?:spike|burst|peak|transient)", 0.15),
        (r"(?:interactive|real.?time|live)\s+(?:stream|traffic|content)", 0.15),
        (r"(?:lower|moderate|medium)\s+(?:bitrate|bandwidth|throughput)", 0.10),
        (r"(?:periodic|irregular)\s+(?:peak|spike|burst)", 0.10),
        (r"(?:twitch|live\s+stream|interactive\s+stream)", 0.15),
        (r"(?:dl|downlink)\s+(?:prb|resource).{0,20}(?:spike|burst|peak)", 0.10),
        (r"(?:chat|interactiv).{0,20}(?:traffic|uplink|bidirection)", 0.05),
    ],
    "youtube": [
        (r"(?:adaptive|abr|variable)\s+(?:bitrate|rate|quality)", 0.20),
        (r"(?:streaming|playback|video)\s+(?:pattern|traffic|session)", 0.15),
        (r"(?:buffer|rebuffer|prefetch)\s+(?:refill|cycle|pattern)", 0.15),
        (r"(?:sustained|consistent|moderate)\s+(?:downlink|dl|download|stream|throughput)", 0.15),
        (r"(?:youtube|video\s+stream|media\s+stream)", 0.10),
        (r"(?:dl|downlink)\s+(?:prb|resource)\s+(?:utilization|allocation).{0,20}(?:spike|periodic|fluctuat)", 0.10),
        (r"(?:unidirection|one.?way|passive|view|watch)", 0.10),
        (r"(?:rx_bytes|received\s+bytes|download).{0,20}(?:moderate|sustained)", 0.05),
    ],
}

_ACTIVITY_NEGATIVE: Dict[str, List[tuple]] = {
    "file": [
        (r"(?:streaming|playback|live\s+stream|adaptive\s+bitrate|abr)", 0.20),
        (r"(?:interactive|real.?time|bursty\s+pattern)", 0.15),
        (r"(?:youtube|twitch)", 0.15),
    ],
    "twitch": [
        (r"(?:file|bulk)\s+(?:transfer|download|upload)", 0.15),
        (r"(?:adaptive\s+bitrate|abr|buffer\s+refill)", 0.15),
        (r"(?:sustained|continuous|steady).{0,15}(?:upload|transmis)", 0.15),
        (r"(?:youtube)", 0.10),
    ],
    "youtube": [
        (r"(?:file|bulk)\s+(?:transfer|download|upload)", 0.15),
        (r"(?:interactive|real.?time|live\s+stream)", 0.15),
        (r"(?:tx_bytes|upload).{0,15}(?:exceed|domin|higher)", 0.15),
        (r"(?:twitch)", 0.10),
    ],
}


def score_activity_reasoning(think_text: str, ground_truth: str) -> float:
    """Score activity classification reasoning. Returns 0.0-1.0."""
    gt = ground_truth.lower().strip()
    if "youtube" in gt or "yt" in gt:
        gt_class = "youtube"
    elif "twitch" in gt:
        gt_class = "twitch"
    elif "file" in gt or "download" in gt:
        gt_class = "file"
    else:
        return 0.0

    pos_patterns = _ACTIVITY_POSITIVE.get(gt_class, [])
    neg_patterns = _ACTIVITY_NEGATIVE.get(gt_class, [])
    t = think_text.lower()

    pos_total = sum(w for _, w in pos_patterns) if pos_patterns else 1.0
    pos_earned = sum(w for pat, w in pos_patterns if re.search(pat, t))
    pos_score = pos_earned / pos_total if pos_total > 0 else 0.0

    neg_total = sum(w for _, w in neg_patterns) if neg_patterns else 1.0
    neg_earned = sum(w for pat, w in neg_patterns if re.search(pat, t)) if neg_patterns else 0.0
    neg_penalty = neg_earned / neg_total if neg_total > 0 else 0.0

    score = 0.6 * pos_score + 0.4 * (1.0 - neg_penalty)
    return max(0.0, min(1.0, score))


# ── Anomaly bounds reasoning quality reward ─────────────────────────
# Temporal localization: start/end indices of anomaly window.
#
# Uses anomaly_type from extra_info to verify:
#   1. Correct affected KPIs are mentioned
#   2. Correct symptom/temporal patterns for that anomaly type
#   3. Timestep references are consistent with the GT window:
#      - "normal at index X" where X is outside GT window → reward
#      - "anomaly/change at index X" where X is inside GT window → reward
#      - Onset index near GT start → reward
#      - Recovery index near GT end → reward
#
# From cold-start CoT (758 traces):
#   "Sudden 40% drop at index 3 → potential anomaly onset"
#   "Consistent deviation persists through index 126"
#   "Sharp recovery at index 127 → return to steady-state"
#   "Post-127 KPIs align with pre-anomaly trends"

def _parse_gt_bounds(ground_truth: str):
    """Parse ground truth 'start, end' into (int, int) or (None, None)."""
    m = re.match(r"(\d+)\s*,\s*(\d+)", ground_truth.strip().strip("'\""))
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def _extract_index_refs(text: str) -> list:
    """Extract (index, context_word) pairs from reasoning text.

    Returns list of (int, str) where str is 'normal'/'anomaly'/'onset'/'recovery'/'neutral'.
    """
    refs = []

    # "normal/stable/baseline at/from index X" → normal
    for m in re.finditer(
        r"(?:normal|stable|baseline|steady|nominal|pre.?anomaly|healthy)"
        r".{0,30}(?:index|timestep|step|t\s*=)\s*(\d+)", text
    ):
        refs.append((int(m.group(1)), "normal"))

    # "at index X ... normal/stable/baseline"
    for m in re.finditer(
        r"(?:index|timestep|step|t\s*=)\s*(\d+)"
        r".{0,30}(?:normal|stable|baseline|steady|nominal|healthy)", text
    ):
        refs.append((int(m.group(1)), "normal"))

    # "anomaly/deviation/degradation at index X" → anomaly
    for m in re.finditer(
        r"(?:anomal|deviat|degradat|abnormal|irregular|instabil|disrupt)"
        r".{0,30}(?:index|timestep|step|t\s*=)\s*(\d+)", text
    ):
        refs.append((int(m.group(1)), "anomaly"))

    # "at index X ... drop/spike/surge/deviation" → anomaly
    for m in re.finditer(
        r"(?:index|timestep|step|t\s*=)\s*(\d+)"
        r".{0,40}(?:drop|spike|surge|deviat|degrad|abnormal|plummet|collaps)", text
    ):
        refs.append((int(m.group(1)), "anomaly"))

    # "onset/begins/starts at index X" → onset
    for m in re.finditer(
        r"(?:onset|begins?|starts?|trigger|initiat)"
        r".{0,20}(?:index|timestep|step|t\s*=)\s*(\d+)", text
    ):
        refs.append((int(m.group(1)), "onset"))

    # "recover/stabilize/return at/after index X" → recovery
    for m in re.finditer(
        r"(?:recover|return|stabiliz|normaliz|resolv|restor|resumes?)"
        r".{0,20}(?:index|timestep|step|t\s*=|after)\s*(\d+)", text
    ):
        refs.append((int(m.group(1)), "recovery"))

    # Generic index references (no clear context)
    for m in re.finditer(r"(?:index|timestep|step|t\s*=)\s*(\d+)", text):
        idx = int(m.group(1))
        if not any(r[0] == idx for r in refs):
            refs.append((idx, "neutral"))

    return refs


def score_anomaly_bounds_reasoning(think_text: str, ground_truth: str = "",
                                   anomaly_type: int = 0) -> float:
    """Score anomaly bounds reasoning using GT window and anomaly type.

    Components (each 0.0-1.0, weighted):
      0.25  Timestep consistency: referenced indices align with GT window
      0.20  Onset accuracy: onset index near GT start
      0.20  Recovery accuracy: recovery index near GT end
      0.15  Affected KPI mentions: correct KPIs for this anomaly type
      0.20  Symptom patterns: correct temporal pattern for this anomaly type

    Returns 0.0-1.0.
    """
    t = think_text.lower()
    gt_start, gt_end = _parse_gt_bounds(ground_truth)

    scores = {}

    # ── 1. Timestep consistency (0.25) ──────────────────────────────
    # Check if the model labels indices correctly: "normal" outside GT
    # window, "anomaly" inside GT window.
    idx_refs = _extract_index_refs(t)
    if idx_refs and gt_start is not None:
        correct = 0
        total_refs = 0
        for idx, label in idx_refs:
            if label in ("normal", "anomaly"):
                total_refs += 1
                inside = gt_start <= idx <= gt_end
                if (label == "anomaly" and inside) or (label == "normal" and not inside):
                    correct += 1
        if total_refs > 0:
            scores["timestep"] = correct / total_refs
        else:
            # Has index references but no labeled ones — partial credit
            scores["timestep"] = 0.3 if len(idx_refs) >= 2 else 0.1
    else:
        scores["timestep"] = 0.0

    # ── 2. Onset accuracy (0.20) ────────────────────────────────────
    # Reward if any "onset" index is within ±5 of GT start.
    onset_indices = [idx for idx, label in idx_refs if label == "onset"]
    if onset_indices and gt_start is not None:
        best_dist = min(abs(idx - gt_start) for idx in onset_indices)
        if best_dist == 0:
            scores["onset"] = 1.0
        elif best_dist <= 3:
            scores["onset"] = 0.8
        elif best_dist <= 5:
            scores["onset"] = 0.6
        elif best_dist <= 10:
            scores["onset"] = 0.3
        else:
            scores["onset"] = 0.1  # at least identified an onset
    elif re.search(r"(?:onset|begins?|starts?|trigger).{0,20}(?:at|from|around|index|timestep)", t):
        scores["onset"] = 0.15  # mentions onset concept but no parseable index
    else:
        scores["onset"] = 0.0

    # ── 3. Recovery accuracy (0.20) ─────────────────────────────────
    # Reward if any "recovery" index is within ±5 of GT end.
    recovery_indices = [idx for idx, label in idx_refs if label == "recovery"]
    if recovery_indices and gt_end is not None:
        best_dist = min(abs(idx - gt_end) for idx in recovery_indices)
        if best_dist == 0:
            scores["recovery"] = 1.0
        elif best_dist <= 3:
            scores["recovery"] = 0.8
        elif best_dist <= 5:
            scores["recovery"] = 0.6
        elif best_dist <= 10:
            scores["recovery"] = 0.3
        else:
            scores["recovery"] = 0.1
    elif re.search(r"(?:recover|return|stabiliz|normaliz|resolv).{0,20}(?:at|after|by|around|index)", t):
        scores["recovery"] = 0.15
    else:
        scores["recovery"] = 0.0

    # ── 4. Affected KPI mentions (0.15) ─────────────────────────────
    # Check if the model mentions KPIs affected by this anomaly type.
    if anomaly_type > 0:
        anomaly_name = reverse_anomalies_type_2_id.get(anomaly_type, "")
        affected_kpis = anomaly_type_2_affected_kpis.get(anomaly_name, [])
        if affected_kpis:
            kpi_mentioned = count_kpi_mentions(t, affected_kpis)
            # Reward for mentioning ≥40% of affected KPIs
            scores["kpi"] = min(1.0, kpi_mentioned / max(1, len(affected_kpis) * 0.4))
        else:
            scores["kpi"] = 0.0
    else:
        # Unknown anomaly type — reward any KPI discussion
        any_kpi = re.findall(
            r"(?:rsrp|bler|mcs|snr|prb|buffer|bytes|throughput|packet)", t
        )
        scores["kpi"] = min(1.0, len(set(any_kpi)) / 3.0)

    # ── 5. Symptom patterns (0.20) ──────────────────────────────────
    # Check if the model uses the right temporal pattern for this anomaly.
    if anomaly_type in _AD_SYMPTOM_PATTERNS:
        sym_patterns = _AD_SYMPTOM_PATTERNS[anomaly_type]
        sym_total = sum(w for _, w in sym_patterns)
        sym_earned = sum(w for pat, w in sym_patterns if re.search(pat, t))
        scores["symptom"] = sym_earned / sym_total if sym_total > 0 else 0.0
    else:
        # Fallback: reward any temporal transition language
        has_transition = bool(re.search(
            r"(?:drop|spike|surge|declin|increas|decreas|oscillat|degradat)", t
        ))
        scores["symptom"] = 0.5 if has_transition else 0.0

    # ── Weighted combination ────────────────────────────────────────
    weights = {
        "timestep": 0.25,
        "onset": 0.20,
        "recovery": 0.20,
        "kpi": 0.15,
        "symptom": 0.20,
    }
    final = sum(weights[k] * scores.get(k, 0.0) for k in weights)
    return max(0.0, min(1.0, final))


# Format reward
def _non_english_ratio(text: str) -> float:
    """Fraction of characters from non-Latin scripts (excluding whitespace).

    Detects language drift (Chinese, Arabic, Cyrillic, etc.) that Qwen3
    is prone to during reasoning. Ignores ASCII and Latin-1 Supplement
    (which includes accented Latin chars like é, ñ, math symbols like ±, °,
    µ, superscripts, smart quotes, em dashes) — these are acceptable in
    English technical writing.
    """
    if not text:
        return 0.0
    visible = [c for c in text if not c.isspace()]
    if not visible:
        return 0.0

    def _is_non_latin(cp: int) -> bool:
        # CJK Unified Ideographs (Chinese/Japanese kanji)
        if 0x4E00 <= cp <= 0x9FFF: return True
        if 0x3400 <= cp <= 0x4DBF: return True  # CJK Extension A
        # Hiragana / Katakana (Japanese)
        if 0x3040 <= cp <= 0x30FF: return True
        # Hangul (Korean)
        if 0xAC00 <= cp <= 0xD7AF: return True
        if 0x1100 <= cp <= 0x11FF: return True
        # Cyrillic (Russian, etc.)
        if 0x0400 <= cp <= 0x04FF: return True
        # Arabic
        if 0x0600 <= cp <= 0x06FF: return True
        if 0x0750 <= cp <= 0x077F: return True
        # Hebrew
        if 0x0590 <= cp <= 0x05FF: return True
        # Thai
        if 0x0E00 <= cp <= 0x0E7F: return True
        # Devanagari (Hindi)
        if 0x0900 <= cp <= 0x097F: return True
        # Greek (except common math symbols which are usually standalone)
        if 0x0370 <= cp <= 0x03FF: return True
        return False

    non_latin = sum(1 for c in visible if _is_non_latin(ord(c)))
    return non_latin / len(visible)


def format_reward(solution_str: str) -> float:
    """Reward for correct output structure (0.0 to 1.0).

    The expected format is:
        <think>\\n{reasoning}\\n</think>\\n\\n{answer with 'quoted' conclusion}

    Broken into granular steps so the model can discover the format
    incrementally — easiest tokens first:

      Step 1: <think> appears at the start of the response    → +0.15
      Step 2: \\n immediately after <think>                    → +0.05
      Step 3: </think> appears somewhere after <think>         → +0.15
      Step 4: \\n before </think>                              → +0.05
      Step 5: reasoning between tags is >50 chars              → +0.20
      Step 6: \\n\\n after </think> (proper spacing)            → +0.05
      Step 7: quoted answer after </think> ('answer')          → +0.15
      Step 8: reasoning is >200 chars (substantial thinking)   → +0.20

    Non-English penalty: a multiplicative factor in [0, 1] is applied
    based on the fraction of non-ASCII characters in the response. Pure
    English text gets 1.0 (no penalty); responses with >10% non-ASCII
    characters are penalized proportionally, reaching 0.0 at 30%+.

    Returns 0.0 to 1.0.
    """
    score = 0.0
    s = solution_str

    # Step 1: <think> at the start of the response (most important to discover)
    stripped = s.lstrip()
    if stripped.startswith("<think>"):
        score += 0.15

    # Step 2: \n immediately after <think>
    if "<think>\n" in s:
        score += 0.05

    # Step 3: </think> appears somewhere
    has_close = "</think>" in s
    if has_close:
        score += 0.15

    # Step 4: \n before </think>
    if "\n</think>" in s:
        score += 0.05

    # Step 5: reasoning between tags >50 chars
    think_match = re.search(r"<think>(.*?)</think>", s, re.DOTALL)
    reasoning_len = len(think_match.group(1).strip()) if think_match else 0
    if reasoning_len > 50:
        score += 0.20

    # Step 6: \n\n after </think>
    if "</think>\n\n" in s or "</think>\n \n" in s:
        score += 0.05

    # Step 7: quoted answer after </think>
    if has_close:
        after_think = s.split("</think>")[-1]
        if re.search(r"'[^']+'\s*\.?\s*$", after_think.strip()):
            score += 0.15

    # Step 8: substantial reasoning (>200 chars) — bonus for longer thinking
    if reasoning_len > 200:
        score += 0.20

    score = min(1.0, score)

    # Non-English penalty: multiplicative factor based on non-Latin script
    # fraction. <10% → no penalty; linear decay to 0 at ≥30%.
    non_latin = _non_english_ratio(s)
    if non_latin > 0.10:
        penalty_factor = max(0.0, 1.0 - (non_latin - 0.10) / 0.20)
        score *= penalty_factor

    return score


# Main entry point
def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    Main scoring function for TelecomTS QA dataset.

    Returns: dict with 'score' (float 0-1) and per-task accuracy fields
    for wandb tracking.
    """
    # Get task type early (needed for per-task metrics)
    task_type = None
    if extra_info and isinstance(extra_info, dict):
        task_type = extra_info.get("question_category")

    result = {"acc": 0.0}

    # Gibberish detection (before extraction)
    if is_gibberish(solution_str):
        result["score"] = 0.0
        return result

    # Extract the final answer
    answer = extract_solution(solution_str)

    # Refusal detection
    if answer and is_refusal(answer):
        result["score"] = 0.0
        return result

    # Format bonus
    fmt_bonus = format_reward(solution_str)

    # Normalize ground truth
    if isinstance(ground_truth, dict):
        if "ground_truth" in ground_truth:
            gt_str = str(ground_truth["ground_truth"])
        elif "target" in ground_truth:
            targets = ground_truth["target"]
            gt_str = targets[0] if isinstance(targets, list) and targets else str(targets)
        elif "answer" in ground_truth:
            gt_str = str(ground_truth["answer"])
        else:
            gt_str = str(ground_truth)
    else:
        gt_str = str(ground_truth)

    # Route to scorer (answer=None → task_score=0, still reward structure/reasoning)
    task_score = 0.0
    if answer:
        if task_type == "anomaly_detection":
            task_score = score_anomaly_detection(answer, gt_str)
        elif task_type == "motion":
            task_score = score_motion(answer, gt_str)
        elif task_type == "jam":
            task_score = score_jam(answer, gt_str)
        elif task_type == "zone":
            task_score = score_zone(answer, gt_str)
        elif task_type == "activity":
            task_score = score_activity(answer, gt_str)
        elif task_type == "root_cause":
            task_score = score_root_cause(answer, gt_str, full_response=solution_str)
        elif task_type == "cong":
            task_score = score_cong(answer, gt_str, full_response=solution_str)
        elif task_type == "anomaly_bounds":
            task_score = score_anomaly_bounds(answer, gt_str)
        else:
            task_score = 1.0 if answer.lower().strip() == gt_str.lower().strip() else 0.0

    # Reasoning quality (0.0-1.0)
    reasoning_score = 0.0
    think_match = re.search(r"<think>(.*?)</think>", solution_str, re.DOTALL)
    if think_match:
        think_text = think_match.group(1)
        anomaly_type = extra_info.get("anomaly_type", 0) if extra_info and isinstance(extra_info, dict) else 0

        if task_type == "zone":
            reasoning_score = score_zone_reasoning(think_text, gt_str)
        elif task_type == "root_cause":
            reasoning_score = score_root_cause_reasoning(think_text, gt_str)
        elif task_type == "anomaly_detection":
            reasoning_score = score_anomaly_detection_reasoning(think_text, gt_str, anomaly_type)
        elif task_type == "motion":
            reasoning_score = score_motion_reasoning(think_text, gt_str)
        elif task_type == "cong":
            reasoning_score = score_cong_reasoning(think_text, gt_str)
        elif task_type == "activity":
            reasoning_score = score_activity_reasoning(think_text, gt_str)
        elif task_type == "anomaly_bounds":
            reasoning_score = score_anomaly_bounds_reasoning(think_text, gt_str, anomaly_type)

    # Structure score (0.0 to 1.0, partial credit for incremental discovery)
    structure_score = fmt_bonus  # format_reward now returns 0.0-1.0

    # Combine: weighted sum of three components, each 0.0-1.0
    #
    #   correctness (0/1):    Did the model get the right answer?
    #   reasoning (0.0-1.0):  Quality of KPI-based reasoning in <think> block
    #   structure (0.0-1.0):  Format compliance (partial credit for each part)
    #
    # Cold-start model already produces <think>...</think>, so structure
    # weight is low — focus reward on correctness and reasoning quality.
    #
    #   correctness: 0.50
    #   reasoning:   0.30
    #   structure:   0.20
    #
    final_score = (
        0.50 * task_score
        + 0.30 * reasoning_score
        + 0.20 * structure_score
    )

    # Populate per-task accuracy
    if task_type == "anomaly_bounds":
        # Use binary vector F1 to match evaluate_tsllm.py
        def _extract_pair(text):
            t = text.strip().strip("'\"")
            m = re.match(r"(\d+)\s*,\s*(\d+)", t)
            if m:
                return int(m.group(1)), int(m.group(2))
            nums = re.findall(r"\d+", t)
            if len(nums) >= 2:
                return int(nums[0]), int(nums[1])
            return None, None
        pred_s, pred_e = _extract_pair(answer) if answer else (None, None)
        truth_s, truth_e = _extract_pair(gt_str)
        if pred_s is not None and truth_s is not None:
            import numpy as np
            vec_gt = np.zeros(128, dtype=int)
            vec_pred = np.zeros(128, dtype=int)
            vec_gt[min(truth_s, truth_e):max(truth_s, truth_e)+1] = 1
            vec_pred[min(pred_s, pred_e):max(pred_s, pred_e)+1] = 1
            tp = (vec_gt & vec_pred).sum()
            prec = tp / vec_pred.sum() if vec_pred.sum() > 0 else 0.0
            rec = tp / vec_gt.sum() if vec_gt.sum() > 0 else 0.0
            task_acc = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        else:
            task_acc = 0.0
    else:
        task_acc = 1.0 if task_score > 0 else 0.0
    result["score"] = final_score
    result["acc"] = task_acc

    return result
