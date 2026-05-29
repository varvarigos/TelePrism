from pathlib import Path
from typing import List, Optional, Dict, Tuple

def normalize_target(long_form: str, category: str, reverse_map: Dict[str, Dict[str, str]]) -> Optional[str]:
    """
    Convert a long-form answer to its normalized class label using a reverse map.

    Args:
        long_form (str): Human-readable label (e.g. "User is watching a twitch stream.")
        category (str): QA type (e.g. "activity")
        reverse_map (dict): Dictionary mapping long-form labels to class labels.

    Returns:
        Optional[str]: Normalized class label (e.g. "twitch"), or None if not found.
    """
    return reverse_map[category].get(long_form.strip(), None)



# -----------------------------------------------------------------------------
# Question-Answer Templates for Each QA Category
# -----------------------------------------------------------------------------


QA_templates: Dict[str, Dict[str, List[Tuple[str, str]]]] = {
    "activity": {
        "User is streaming youtube.": [
            ("Which of the following was the user doing: streaming YouTube, watching Twitch, or downloading a file?", "The user was streaming YouTube."),
            ("What was the user's activity, YouTube, Twitch, or File download?", "The user was streaming YouTube."),
            ("Identify the service in use: YouTube, Twitch, or File download.", "YouTube was the service being used."),
            ("Was the user streaming YouTube, Twitch, or downloading a File?", "The user was streaming YouTube."),
            ("Which type of activity occurred, YouTube viewing, Twitch stream, or file transfer?", "YouTube streaming was the activity."),
            ("What application was active, YouTube, Twitch, or File download?", "YouTube was the primary application used."),
            ("From YouTube, Twitch, or File download, which was active?", "YouTube was active."),
            ("Which of the three traffic types is observed here: YouTube, Twitch, or File download?", "YouTube streaming was observed."),
            ("What traffic pattern does this segment match: YouTube, Twitch, or File download?", "This segment matches YouTube streaming."),
            ("Choose the user's activity: YouTube streaming, Twitch viewing, or downloading a File.", "The user was actively watching YouTube."),
        ],
        "User is watching a twitch stream.": [
            ("Which of the following was the user doing: streaming YouTube, watching Twitch, or downloading a file?", "The user was watching a Twitch stream."),
            ("What was the user's activity, YouTube, Twitch, or File download?", "Twitch traffic was observed during this window."),
            ("Identify the service in use: YouTube, Twitch, or File download.", "This is a segment from a Twitch session."),
            ("Was the user streaming YouTube, Twitch, or downloading a File?", "The user was engaged in Twitch streaming."),
            ("Which type of activity occurred, YouTube viewing, Twitch stream, or file transfer?", "Twitch streaming was the activity."),
            ("What application was active, YouTube, Twitch, or File download?", "Twitch was the primary application used."),
            ("From YouTube, Twitch, or File download, which was active?", "Twitch was active."),
            ("Which of the three traffic types is observed here: YouTube, Twitch, or File download?", "Twitch content was being streamed."),
            ("What traffic pattern does this segment match: YouTube, Twitch, or File download?", "This session involves Twitch streaming."),
            ("Choose the user's activity: YouTube streaming, Twitch viewing, or downloading a file.", "The user was watching a Twitch stream."),
        ],
        "User is downloading a large file.": [
            ("Which of the following was the user doing: streaming YouTube, watching Twitch, or downloading a file?", "The user was downloading a file."),
            ("What was the user's activity, YouTube, Twitch, or File download?", "This trace indicates a file download."),
            ("Identify the service in use: YouTube, Twitch, or File download.", "This is a segment from a file download session."),
            ("Was the user streaming YouTube, Twitch, or downloading a File?", "The user was downloading a file."),
            ("Which type of activity occurred, YouTube viewing, Twitch stream, or file transfer?", "File download was the activity."),
            ("What application was active, YouTube, Twitch, or File download?", "File download was the primary application used."),
            ("From YouTube, Twitch, or File download, which was active?", "File download was active."),
            ("Which of the three traffic types is observed here: YouTube, Twitch, or File download?", "File download traffic was detected."),
            ("What traffic pattern does this segment match: YouTube, Twitch, or File download?", "This segment matches file download traffic."),
            ("Choose the user's activity: YouTube streaming, Twitch viewing, or downloading a file.", "The user was downloading a file."),
        ],
    },

    "zone": {
        "User is static at zone 1.": [
            ("Which zone was the user in, Zone A, Zone B, or Zone C?", "The user was in Zone A."),
            ("What test area was active: Zone A, B, or C?", "This trace corresponds to Zone A."),
            ("Can you identify the zone, A, B, or C?", "Zone A was the user's location."),
            ("From Zone A, Zone B, or Zone C, where was the user?", "This session took place in Zone A."),
            ("Was this activity in Zone A, B, or C?", "Zone A was the location of the user."),
            ("Which deployment zone applies here, A, B, or C?", "Traffic was recorded in Zone A."),
            ("Select the zone: A, B, or C.", "This activity occurred in Zone A."),
            ("Which zone does this sample represent?", "Zone A is the relevant test area."),
            ("Where was the user, Zone A, B, or C?", "The user operated in Zone A."),
            ("What was the user's location, Zone A, Zone B, or Zone C?", "The user was located in Zone A."),
        ],
        "User is static at zone 2.": [
            ("Which zone was the user in, Zone A, Zone B, or Zone C?", "The user was in Zone B."),
            ("What test area was active: Zone A, B, or C?", "This trace corresponds to Zone B."),
            ("Can you identify the zone, A, B, or C?", "Zone B was the user's location."),
            ("From Zone A, Zone B, or Zone C, where was the user?", "This session took place in Zone B."),
            ("Was this activity in Zone A, B, or C?", "Zone B was the location of the user."),
            ("Which deployment zone applies here, A, B, or C?", "Traffic was recorded in Zone B."),
            ("Select the zone: A, B, or C.", "This activity occurred in Zone B."),
            ("Which zone does this sample represent?", "Zone B is the relevant test area."),
            ("Where was the user, Zone A, B, or C?", "The user operated in Zone B."),
            ("What was the user's location, Zone A, Zone B, or Zone C?", "The user was located in Zone B."),
        ],
        "User is static at zone 3.": [
            ("Which zone was the user in, Zone A, Zone B, or Zone C?", "The user was in Zone C."),
            ("What test area was active: Zone A, B, or C?", "This trace corresponds to Zone C."),
            ("Can you identify the zone, A, B, or C?", "Zone C was the user's location."),
            ("From Zone A, Zone B, or Zone C, where was the user?", "This session took place in Zone C."),
            ("Was this activity in Zone A, B, or C?", "Zone C was the location of the user."),
            ("Which deployment zone applies here, A, B, or C?", "Traffic was recorded in Zone C."),
            ("Select the zone: A, B, or C.", "This activity occurred in Zone C."),
            ("Which zone does this sample represent?", "Zone C is the relevant test area."),
            ("Where was the user, Zone A, B, or C?", "The user operated in Zone C."),
            ("What was the user's location, Zone A, Zone B, or Zone C?", "The user was located in Zone C."),
        ],
        "User is in motion.": [
            ("Was the user static at Zone A, B, or C, or in motion?", "The user was in motion."),
            ("Identify the user location, Zone A, Zone B, Zone C, or moving?", "The user was transitioning between zones."),
            ("Where was the user located: Zone A, B, C, or moving?", "The user was moving across zones."),
            ("Was the user assigned to a fixed zone or in motion?", "This trace reflects a user in motion."),
            ("Which zone does this traffic relate to: A, B, C, or motion?", "Movement between zones occurred."),
            ("What was the user's location status, static or mobile?", "The user was mobile."),
            ("Select the user's condition, Zone A, B, C, or moving?", "Zone localization is unclear due to mobility."),
            ("From Zone A/B/C or moving, where was the user?", "The user changed zones during the session."),
            ("Was the user static or in transit?", "The user was in transit."),
            ("Was the user at a specific zone or moving?", "The trace corresponds to cross-zone mobility."),
        ],
    },

    "jam": {
        "There are no jammers.": [
            ("Was the user affected by jamming or was the link clean?", "There were no jammers present."),
            ("Did the network suffer from jamming or operate interference-free?", "The signal was clean with no jamming detected."),
            ("Choose the condition: jamming present or not?", "This is a jammer-free session."),
            ("Was jamming detected, or was the signal unaffected?", "No jamming occurred in this time window."),
            ("Identify the scenario: jamming or no jamming?", "Jamming was not present during this sample."),
            ("What is the condition: jamming or normal operation?", "The link was operating normally."),
            ("Was the environment jammed or interference-free?", "This period was free of jamming activity."),
            ("Select the jamming state: present or absent?", "The transmission was unaffected by jamming."),
            ("Did the user experience jamming or normal signal quality?", "There were no signs of jamming."),
            ("Was this sample impacted by jamming?", "The link operated without jamming."),
        ],
        "There is a jammer.": [
            ("Was the user affected by jamming or was the link clean?", "Jammers were affecting the signal."),
            ("Did the network suffer from jamming or operate interference-free?", "Jamming interference was present."),
            ("Choose the condition: jamming present or not?", "This session involved jamming conditions."),
            ("Was jamming detected, or was the signal unaffected?", "The user experienced signal interference."),
            ("Identify the scenario: jamming or no jamming?", "The signal was disrupted by jamming activity."),
            ("What interference condition applies here: jamming or normal operation?", "The link was impaired by jamming."),
            ("Was the environment jammed or interference-free?", "External jammers influenced the transmission."),
            ("Select the jamming state: present or absent?", "This trace shows signs of jamming."),
            ("Did the user experience jamming or normal signal quality?", "Jamming was detected during this interval."),
            ("Was this sample impacted by jamming?", "The signal was degraded due to active jammers."),
        ],
    },

    "cong": {
        "The network is not congested.": [
            ("Was the network congested or operating normally?", "The network was not congested."),
            ("Choose the condition: congestion or no congestion?", "No congestion was detected."),
            ("Was the network under heavy load or performing well?", "Traffic flowed normally during this period."),
            ("Did the user experience congestion?", "There was no congestion-related degradation."),
            ("Was congestion affecting this session?", "Utilization was within normal bounds."),
            ("Identify the congestion status: normal or overloaded?", "The network was operating normally."),
            ("Was the system congested?", "No, congestion was not observed here."),
            ("Did the network show overload symptoms?", "No signs of overload were present."),
            ("Was network saturation evident?", "This trace shows normal throughput conditions."),
            ("Is this a congested or uncongested session?", "This session was uncongested."),
        ],
        "The network is congested.": [
            ("Was the network congested or operating normally?", "The network was congested during this window."),
            ("Choose the condition: congestion or no congestion?", "Congestion affected performance in this sample."),
            ("Was the network under heavy load or performing well?", "Heavy load was detected in this period."),
            ("Did the user experience congestion?", "Yes, the user experienced congestion."),
            ("Was congestion affecting this session?", "This trace indicates congestion was present."),
            ("Identify the congestion status: normal or overloaded?", "The network was overloaded."),
            ("Was the system congested?", "Yes, the system was congested."),
            ("Did the network show overload symptoms?", "Yes, overload symptoms were present."),
            ("Was network saturation evident?", "Yes, network saturation was evident."),
            ("Is this a congested or uncongested session?", "This session was congested."),
        ],
    },

    "motion": {
        "Yes": [
            ("Was the user stationary or in motion during this window?", "The user was in motion."),
            ("Select the mobility status: moving or stationary?", "Moving was detected."),
            ("Was the user static or moving across locations?", "The user was moving across locations."),
            ("Which condition applies, mobile or static?", "The user was mobile."),
            ("Identify the user state: still or in motion?", "The user was in motion."),
            ("Was the user moving or not?", "The user was moving."),
            ("From static or mobile, which best describes the user?", "The user was mobile during this session."),
            ("Can you classify the user as moving or still?", "This segment reflects a moving user."),
            ("What is the user mobility status: moving or not moving?", "The user was moving."),
            ("Was movement present in this segment?", "Yes, there was movement during this session."),
        ],
        "No": [
            ("Was the user stationary or in motion during this window?", "The user was stationary."),
            ("Select the mobility status: moving or stationary?", "The user was stationary."),
            ("Was the user static or moving across locations?", "The user was static."),
            ("Which condition applies, mobile or static?", "The user was static."),
            ("Identify the user state: still or in motion?", "The user was still."),
            ("Was the user moving or not?", "The user was not moving."),
            ("From static or mobile, which best describes the user?", "This segment reflects a static user."),
            ("Can you classify the user as moving or still?", "The user was still."),
            ("What is the user mobility status: moving or not moving?", "The user was not moving."),
            ("Was movement present in this segment?", "No, there was no movement during this session."),
        ],
    }
}



# -----------------------------------------------------------------------------
# Answer Normalization Mapping: Human-readable → Normalized Labels
# -----------------------------------------------------------------------------

answer_normalization: Dict[str, Dict[str, str]] = {
    "activity": {
        "User is streaming youtube.": "youtube",
        "youtube": "youtube",
        "User is watching a twitch stream.": "twitch",
        "twitch": "twitch",
        "User is downloading a large file.": "file",
        "file": "file",
        "file download": "file"
    },
    "zone": {
        "User is static at zone 1.": "A",
        "zone a": "A",
        "User is static at zone 2.": "B",
        "zone b": "B",
        "User is static at zone 3.": "C",
        "zone c": "C",
        "User is in motion.": "motion"
    },
    "cong": {
        "The network is not congested.": "no",
        "No Congestion": "no",
        "The network is congested.": "yes",
        "Congestion": "yes"
    },
    "motion": {
        "Yes": "mobile",
        "Mobile": "mobile",
        "No": "static",
        "Stationary": "static"
    },
    "jam": {
        "There is a jammer.": "yes",
        "There are no jammers.": "no"
    }
}


# -----------------------------------------------------------------------------
# Anomaly Type to ID Mapping
# -----------------------------------------------------------------------------

anomalies_type_2_id = {
    "Antenna Failure": 1,
    "Co-Channel Interference (Mild)": 2,
    "Co-Channel Interference (Severe)": 3,
    "Faulty RF Filters (Temporal)": 4,
    "Doppler Shift (Severe)": 5,
    "Faulty Handover Algorithm (Too Frequent)": 6,
    "Buffer Overflow (Gradual Buildup)": 7,
    "Resource Allocation Bugs": 8,
    "High Network Congestion (Gradual Buildup)": 9,
    "High Network Congestion (Sudden Spike)": 10,
    "Jamming": 11,
}

reverse_anomalies_type_2_id = {v: k for k, v in anomalies_type_2_id.items()}


# -----------------------------------------------------------------------------
# Anomaly Type to Affected KPIs Mapping
# -----------------------------------------------------------------------------

anomaly_type_2_affected_kpis: dict[str, list[str]] = {
    "Antenna Failure": [
        "Estimated_UL_Buffer",
        "UL_BLER",
        "RX_Bytes",
        "PRBs_UL_Current",
        "UL_SNR",
        "UL_NumberOfPackets",
        "DL_NumberOfPackets",
        "TX_Bytes",
        "DL_MCS"
        "PRBs_DL_Current",
        "DL_BLER",
        "RSRP",
        "UL_MCS",
    ],
    "Co-Channel Interference (Mild)": [
        "PRB_Utilization_UL",
        "UL_BLER",
        "UL_NPRB",
        "RX_Bytes",
        "PRB_Utilization_DL",
        "UL_SNR",
        "TX_Bytes",
        "DL_BLER",
        "RSRP",
    ],
    "Co-Channel Interference (Severe)": [
        "PRB_Utilization_UL",
        "UL_BLER",
        "UL_NPRB",
        "RX_Bytes",
        "PRB_Utilization_DL",
        "UL_SNR",
        "UL_NumberOfPackets",
        "DL_NumberOfPackets",
        "TX_Bytes",
        "DL_BLER",
        "RSRP",
    ],
    "Faulty RF Filters (Temporal)": [
        "PRB_Utilization_UL",
        "UL_BLER",
        "UL_NPRB",
        "RX_Bytes",
        "PRB_Utilization_DL",
        "UL_SNR",
        "TX_Bytes",
        "DL_BLER",
        "RSRP",
    ],
    "Doppler Shift (Severe)": [
        "UL_NPRB",
        "RX_Bytes",
        "UL_SNR",
        "TX_Bytes",
        "DL_MCS",
        "RSRP",
        "UL_MCS",
    ],
    "Faulty Handover Algorithm (Too Frequent)": [
        "Estimated_UL_Buffer",
        "UL_BLER",
        "RX_Bytes",
        "DL_NumberOfPackets",
        "UL_SNR",
        "UL_NumberOfPackets",
        "TX_Bytes",
        "DL_BLER",
        "RSRP",
    ],
    "Buffer Overflow (Gradual Buildup)": [
        "Estimated_UL_Buffer",
        "PRB_Utilization_UL",
        "UL_BLER",
        "UL_NPRB",
        "RX_Bytes",
        "DL_NumberOfPackets",
        "PRB_Utilization_DL",
        "TX_Bytes",
        "DL_BLER",
        "UL_NumberOfPackets",
    ],
    "High Network Congestion (Gradual Buildup)": [
        "Estimated_UL_Buffer",
        "PRB_Utilization_UL",
        "UL_BLER",
        "UL_NPRB",
        "PRB_Utilization_DL",
        "RX_Bytes",
        "DL_NumberOfPackets",
        "TX_Bytes",
        "DL_BLER",
        "UL_NumberOfPackets",
    ],
    "High Network Congestion (Sudden Spike)": [
        "Estimated_UL_Buffer",
        "PRB_Utilization_UL",
        "UL_BLER",
        "UL_NPRB",
        "PRB_Utilization_DL",
        "RX_Bytes",
        "DL_NumberOfPackets",
        "TX_Bytes",
        "DL_BLER",
        "UL_NumberOfPackets",
    ],
    "Resource Allocation Bugs": [
        "PRB_Utilization_UL",
        "UL_BLER",
        "UL_NPRB",
        "PRB_Utilization_DL",
        "PRBs_UL_Current",
        "RX_Bytes",
        "TX_Bytes",
        "PRBs_DL_Current",
        "DL_BLER",
    ],
    "Jamming": [
        "DL_BLER",
        "UL_BLER",
        "UL_MCS",
        "TX_Bytes",
        "RX_Bytes",
    ],
}

# -----------------------------------------------------------------------------
# Anomaly Type to Symptom Description Mapping
# -----------------------------------------------------------------------------

anomaly_type_2_symptom = {
    "Antenna Failure": (
        "Significant drop in reference signal received power (10-20 units). "
        "Uplink SNR fell (5-15 units). Both uplink and downlink block error "
        "rates surged exponentially. Uplink and downlink MCS decreased "
        "(5-15 units). PRBs for downlink (30-80%) and uplink (50-80%) "
        "decreased. Transmitted and received bytes declined logarithmically. "
        "Uplink buffer backlog grew exponentially. Uplink packet count rose "
        "(50-200%), downlink packet count increased (30-100%)."
    ),
    "Co-Channel Interference (Mild)": (
        "Decreased reference signal power (3-8 units), reduced uplink SNR "
        "(1-3 units), increased block error rates (uplink & downlink 10-40%), "
        "reduced transmitted and received bytes (5-15%), and higher PRB "
        "utilization and allocation (5-20%)."
    ),
    "Co-Channel Interference (Severe)": (
        "Significant decreases in reference signal received power (8-15 dB), "
        "uplink signal-to-noise ratio (8-15 dB), transmitted/received bytes "
        "(30-80%), and uplink/downlink packet counts (40-70%); increased "
        "uplink/downlink block error rates (50-200%), PRB utilization "
        "(30-80%), and PRB allocations (20-70%)."
    ),
    "Faulty RF Filters (Temporal)": (
        "Linear decreases in Reference Signal Received Power and Uplink "
        "Signal-to-Noise Ratio; exponential increases in both uplink and "
        "downlink block error rates and PRB utilization; logarithmic decay "
        "in transmitted and received bytes; linear increase in PRBs allocated."
    ),
    "Doppler Shift (Severe)": (
        "Periodic fluctuations in RSRP (3-8 dB) and uplink SNR (2-5 dB) at "
        "2-3 Hz; downlink and uplink MCS decreased by 1-2 levels; multiplicative "
        "oscillations in transmitted/received bytes (factor 0.20-0.40) and PRBs "
        "allocated (factor 0.30-0.60) at 2-3 Hz."
    ),
    "Faulty Handover Algorithm (Too Frequent)": (
        "Periodic fluctuations in Reference Signal Received Power (amplitude "
        "2.00-5.00, freq 0.1000-0.3000 Hz); periodic uplink SNR fluctuations "
        "(amplitude 1.00-3.00, freq 0.1000-0.3000 Hz); increased block error "
        "rates on both uplink and downlink (30%-150%); reduced transmitted "
        "and received byte counts (10%-50% decrease); increased uplink buffer "
        "backlog (50%-200%); increased uplink/downlink packet counts (50%-200%)."
    ),
    "Buffer Overflow (Gradual Buildup)": (
        "Exponential growth in uplink buffer backlog and block error rates; "
        "logarithmic decay of transmitted and received bytes; linear increase "
        "in uplink and downlink packet counts; logistic growth in PRB utilization "
        "and PRBs allocated."
    ),
    "Resource Allocation Bugs": (
        "Downlink and uplink PRBs and their utilizations, allocated PRBs, and "
        "transmitted/received bytes all exhibited multiplicative oscillations "
        "(amplitude 0.30-1.00, frequency 0.3-1.0 Hz). Both uplink and downlink "
        "block error rates increased by 20-100%."
    ),
    "High Network Congestion (Gradual Buildup)": (
        "Exponential growth in uplink buffer backlog and block error rates; "
        "logistic growth in PRB utilization and PRB allocation; logarithmic decay "
        "in both transmitted and received bytes; exponential increase in uplink "
        "and downlink packet counts."
    ),
    "High Network Congestion (Sudden Spike)": (
        "Significant uplink buffer backlog (100-400% increase), elevated PRB "
        "utilization for both uplink and downlink (30-80% increase), higher "
        "block error rates (50-150% increase), reduced transmitted/received bytes "
        "(20-50% decrease), lower packet counts (40-60% decrease), and increased "
        "PRBs allocated (40-100% increase)."
    ),
    "Jamming": (
        "DL_BLER and UL_BLER increase, UL_MCS decreases, and TX_Bytes and "
        "RX_Bytes experience drops."
    ),
}