import re
from datetime import datetime
import numpy as np
import pandas as pd

anomaly_list = [
        "Antenna Failure", "Co-Channel Interference (Mild)",
        "Co-Channel Interference (Severe)", "Faulty RF Filters (Temporal)", "Doppler Shift (Severe)",
        "Faulty Handover Algorithm (Too Frequent)", "Buffer Overflow (Gradual Buildup)", 
        "High Network Congestion (Gradual Buildup)", "High Network Congestion (Sudden Spike)", 
        "Resource Allocation Bugs", "Jamming"
    ]

def anomaly_detection_prompt(sequence, multimodal=False, context=False):
    df = sequence['dataframe']
    start_time = df['timestamp'][0]
    end_time = list(df['timestamp'])[-1]
    df = df.astype(str)
    metric_lines = [f"{col}:" + ' ' + ' '.join(df[col].tolist()) for col in df.columns if col != "timestamp"]
    time_series_data = '\n'.join(metric_lines)
    seq_len = len(df)

    prompt = (
        "You are an AI assistant tasked with analyzing time-series data for anomalies in a wireless network. "
        "You will be provided with a time-series dataset containing various metrics and a specific time range to analyze. "
        "The time-series is sampled every 0.1 seconds (i.e. timestamps are a decisecond apart), and contains a total of {} time steps. "
        "Your goal is to detect any anomalies within this range and identify the timestamps where they occur. "
    ).format(seq_len)

    if context:
        prompt += (
            "Note: Wireless network data is naturally **noisy and erratic**, even under normal conditions. "
            "Sporadic spikes, sharp drops, or momentary fluctuations can appear **without indicating any true anomaly**. "
            "This sequence is only {:.1f} seconds long, so be especially cautious in interpreting short-term changes as significant. "
            "Only mark something as anomalous if there is **clear and sustained evidence** of abnormal behavior across multiple metrics.\n\n"
        ).format(seq_len / 10)

    if not multimodal:
        prompt += "First, review the time-series data provided from time {} to {}:\n\n".format(start_time, end_time)
        prompt += time_series_data
        prompt += "\n\n"

    if context:
        prompt += "To detect anomalies, follow these steps:\n\n"
        prompt += "1. Begin by scanning the time-series for any unusual behavior: sharp spikes or drops, sustained deviations, or values inconsistent with the expected range.\n"
        prompt += "2. Consider inter-metric relationships — for example, whether high buffer utilization coincides with low throughput or high BLER.\n"
        prompt += "3. All anomalies occur at the same timestamp range, so you should identify a single set of timestamps for the anomaly event and attribute affected metrics to that period.\n"

    prompt += (
        "\n\nSummarize your conclusion as follows:\n\n"
        "```\n"
        "<conclusion>\n"
        "Anomaly Detected: [Yes/No]\n"
        "</conclusion>\n"
        "```\n\n"
        "Only base your analysis on the provided time range. If no anomaly is detected, write:\n"
        "```\n"
        "<conclusion>\n"
        "Anomaly Detected: No\n"
        "</conclusion>\n"
        "```\n"
        "Do not include additional comments or summaries outside this format."
    )

    return prompt

def anomaly_boundary_prompt(sequence, multimodal=False):
    df = sequence['dataframe']
    start_time = df['timestamp'][0]
    end_time = list(df['timestamp'])[-1]
    df = df.astype(str)
    metric_lines = [f"{col}:" + ' ' + ' '.join(df[col].tolist()) for col in df.columns if col != "timestamp"]
    time_series_data = '\n'.join(metric_lines)
    seq_len = len(df)

    prompt = (
        "You are an AI assistant tasked with analyzing time-series data for anomalies in a wireless network. "
        "You will be provided with a time-series dataset containing various metrics and a specific time range to analyze. "
        "The time-series is sampled every 0.1 seconds (i.e. timestamps are a decisecond apart), and contains a total of {} time steps. "
        "Your goal is to identify a **single contiguous time interval** during which an anomaly occurs. "
        "There is exactly one anomaly in the data, and it may span the entire sequence or just a sub-segment.\n\n"

    ).format(seq_len, seq_len / 10)
    
    if not multimodal:
        prompt += "First, review the time-series data provided from time {} to {}:\n\n".format(start_time, end_time)
        prompt += time_series_data
        prompt += "\n\n"

    prompt += (
        "Summarize your conclusion as follows:\n\n"
        "```\n"
        "<conclusion>\n"
        "Anomaly Timestamps: (YYYY-MM-DD HH:MM:SS.sss, YYYY-MM-DD HH:MM:SS.sss)\n"
        "</conclusion>\n"
        "```\n\n"
        "Do not include any additional commentary or explanation outside the specified format. Respond with *only* the <conclusion> block and nothing else."
    )

    return prompt

anomaly_descriptions = "Antenna Failure: RSRP decreases by 10.00 to 20.00. UL_SNR decreases by 5.00 to 15.00. DL_BLER experiences exponential growth with rate 0.035 to 0.050. UL_BLER experiences exponential growth with rate 0.035 to 0.045. DL_MCS decreases by 5.00 to 15.00. UL_MCS decrease by 5.00 to 15.00. PRBs_DL_Current decreases by 30.00% to 80.00%. PRBs_UL_Current decreases by 50.00% to 80.00%. TX_Bytes follows a logarithmic decay pattern with factor 0.200 to 0.30. RX_Bytes follows a logarithmic decay pattern with factor 0.200 to 0.30. Estimated_UL_Buffer experiences exponential growth with rate 0.060 to 0.075. UL_NumberOfPackets increases by 50.00% to 200.00%. DL_NumberOfPackets increases by 30.00% to 100.00%. \nCo-Channel Interference (Mild): RSRP decreases by 3.00 to 8.00. UL_SNR decreases by 1.00 to 3.00. DL_BLER increases by 10.00% to 40.00%. UL_BLER increases by 10.00% to 40.00%. TX_Bytes decreases by 5.00% to 15.00%. RX_Bytes decreases by 5.00% to 15.00%. PRB_Utilization_DL increases by 5.00% to 20.00%. PRB_Utilization_UL increases by 5.00% to 20.00%. UL_NPRB increases by 5.00% to 20.00%.\nCo-Channel Interference (Severe): RSRP decreases by 8.00 to 15.00. UL_SNR decreases by 8.00 to 15.00. DL_BLER increases by 50.00% to 200.00%. UL_BLER increases by 50.00% to 200.00%. TX_Bytes decreases by 30.00% to 80.00%. RX_Bytes decreases by 30.00% to 80.00%. PRB_Utilization_DL increases by 30.00% to 80.00%. PRB_Utilization_UL increases by 30.00% to 80.00%. UL_NumberOfPackets decreases by 40.00% to 70.00%. DL_NumberOfPackets decreases by 40.00% to 70.00%. UL_NPRB increases by 20.00% to 70.00%.\nFaulty RF Filters (Temporal): RSRP decreases linearly by 0.20 to 0.50 per time step. UL_SNR decreases linearly by 0.20 to 0.35 per time step. DL_BLER experiences exponential growth with rate 0.025 to 0.035. UL_BLER experiences exponential growth with rate 0.020 to 0.030. TX_Bytes follows a logarithmic decay pattern with factor 0.012 to 0.20. RX_Bytes follows a logarithmic decay pattern with factor 0.012 to 0.20. PRB_Utilization_DL experiences exponential growth with rate 0.006 to 0.009. PRB_Utilization_UL experiences exponential growth with rate 0.006 to 0.009. UL_NPRB decreases linearly by 0.02 to 0.05 per time step.\nDoppler Shift (Severe): RSRP fluctuates periodically with amplitude 3.00 to 8.00 and frequency 2.0000 to 3.0000 Hz. UL_SNR fluctuates periodically with amplitude 2.00 to 5.00 and frequency 2.0000 to 3.0000 Hz. DL_MCS decreases by 1.00 to 2.00. UL_MCS decreases by 1.00 to 2.00. TX_Bytes oscillates multiplicatively with amplitude factor 0.20 to 0.40 and frequency 2.0000 to 3.0000 Hz. RX_Bytes oscillates multiplicatively with amplitude factor 0.20 to 0.40 and frequency 2.0000 to 3.0000 Hz. UL_NPRB oscillates multiplicatively with amplitude factor 0.30 to 0.60 and frequency 2.0000 to 3.0000 Hz.\nFaulty Handover Algorithm (Too Frequent): RSRP fluctuates periodically with amplitude 2.00 to 5.00 and frequency 0.1000 to 0.3000 Hz. UL_SNR fluctuates periodically with amplitude 1.00 to 3.00 and frequency 0.1000 to 0.3000 Hz. DL_BLER increases by 30.00% to 150.00%. UL_BLER increases by 30.00% to 150.00%. TX_Bytes decreases by 10.00% to 50.00%. RX_Bytes decreases by 10.00% to 50.00%. Estimated_UL_Buffer increases by 50.00% to 200.00%. UL_NumberOfPackets increases by 50.00% to 200.00%. DL_NumberOfPackets increases by 50.00% to 200.00%.\nBuffer Overflow (Gradual Buildup): Estimated_UL_Buffer experiences exponential growth with rate 0.120 to 0.200. TX_Bytes follows a logarithmic decay pattern with factor 0.170 to 0.20. RX_Bytes follows a logarithmic decay pattern with factor 0.150 to 0.18. UL_NumberOfPackets decreases linearly by 5.00 to 10.00 per time step. DL_NumberOfPackets decreases linearly by 5.00 to 10.00 per time step. UL_BLER experiences exponential growth with rate 0.030 to 0.040. DL_BLER experiences exponential growth with rate 0.030 to 0.040. PRB_Utilization_DL experiences logistic growth with rate 0.070 to 0.100. PRB_Utilization_UL experiences logistic growth with rate 0.060 to 0.080. UL_NPRB experiences logistic growth with rate 0.050 to 0.080.\nResource Allocation Bugs: PRBs_DL_Current oscillates multiplicatively with amplitude factor 0.50 to 1.00 and frequency 0.3000 to 1.0000 Hz. PRBs_UL_Current oscillates multiplicatively with amplitude factor 0.50 to 1.00 and frequency 0.3000 to 1.0000 Hz. PRB_Utilization_DL oscillates multiplicatively with amplitude factor 0.30 to 1.00 and frequency 0.3000 to 1.0000 Hz. PRB_Utilization_UL oscillates multiplicatively with amplitude factor 0.30 to 1.00 and frequency 0.3000 to 1.0000 Hz. UL_BLER increases by 20.00% to 100.00%. DL_BLER increases by 20.00% to 100.00%. TX_Bytes oscillates multiplicatively with amplitude factor 0.40 to 0.70 and frequency 0.3000 to 1.0000 Hz. RX_Bytes oscillates multiplicatively with amplitude factor 0.40 to 0.70 and frequency 0.3000 to 1.0000 Hz. UL_NPRB oscillates multiplicatively with amplitude factor 0.40 to 0.90 and frequency 0.3000 to 1.0000 Hz.\nHigh Network Congestion (Gradual Buildup): Estimated_UL_Buffer experiences exponential growth with rate 0.050 to 0.130. PRB_Utilization_DL experiences logistic growth with rate 0.040 to 0.070. PRB_Utilization_UL experiences logistic growth with rate 0.030 to 0.060. UL_BLER experiences exponential growth with rate 0.017 to 0.025. DL_BLER experiences exponential growth with rate 0.017 to 0.025. TX_Bytes follows a logarithmic decay pattern with factor 0.110 to 0.14. RX_Bytes follows a logarithmic decay pattern with factor 0.100 to 0.12. UL_NumberOfPackets experiences exponential growth with rate 0.080 to 0.130. DL_NumberOfPackets experiences exponential growth with rate 0.080 to 0.130. UL_NPRB experiences logistic growth with rate 0.030 to 0.070.\nHigh Network Congestion (Sudden Spike): Estimated_UL_Buffer increases by 100.00% to 400.00%. PRB_Utilization_DL increases by 30.00% to 80.00%. PRB_Utilization_UL increases by 30.00% to 80.00%. UL_BLER increases by 50.00% to 150.00%. DL_BLER increases by 50.00% to 150.00%. TX_Bytes decreases by 20.00% to 50.00%. RX_Bytes decreases by 20.00% to 50.00%. UL_NumberOfPackets decreases by 40.00% to 60.00%. DL_NumberOfPackets decreases by 40.00% to 60.00%. UL_NPRB increases by 40.00% to 100.00%.\n\n"

def root_cause_prompt(sequence, multimodal=False, descriptions = False):
    df = sequence['dataframe']
    start_time = df['timestamp'][0]
    end_time = list(df['timestamp'])[-1]
    df = df.astype(str)
    metric_lines = [f"{col}:" + ' ' + ' '.join(df[col].tolist()) for col in df.columns if col != "timestamp"]
    time_series_data = '\n'.join(metric_lines)
    seq_len = len(df)

    anomaly_list = [
        "Antenna Failure", "Co-Channel Interference (Mild)",
        "Co-Channel Interference (Severe)", "Faulty RF Filters (Temporal)", "Doppler Shift (Severe)",
        "Faulty Handover Algorithm (Too Frequent)", "Buffer Overflow (Gradual Buildup)", 
        "High Network Congestion (Gradual Buildup)", "High Network Congestion (Sudden Spike)", 
        "Resource Allocation Bugs"
    ]

    prompt = (
        "You are an AI assistant tasked with diagnosing a known anomaly in wireless network time-series data. "
        "You will be provided with a short time-series segment sampled every 0.1 seconds, covering {:.1f} seconds and {} time steps. "
        "This sequence ranges from {} to {} and **is confirmed to contain an anomaly**.\n\n"
    ).format(seq_len / 10, seq_len, start_time, end_time)

    prompt += (
        "Your task is to identify the most plausible anomaly type **from the following list**:\n\n"
        "{}\n\n"
        "Please analyze the metrics below and select the **single most likely anomaly**.\n\n"
    ).format(', '.join(anomaly_list))

    if descriptions == True:
        prompt += "Here is a summary on how the provided anomalies generally behave:\n\n"
        prompt += anomaly_descriptions

    if not multimodal:
        prompt += "Here is the time-series data:\n\n" + time_series_data + "\n\n"
    
    prompt += (
        "Summarize your conclusions as follows:\n\n"
        "```\n"
        "<conclusion>\n"
        "Anomaly Type: [One exact string from the predefined anomaly list.]\n"
        "</conclusion>\n"
        "```\n\n"
        "Do not include any additional commentary or explanation outside the specified format. Respond with *only* the <conclusion> block and nothing else."
    )

    return prompt

def _strip_think(text: str) -> str:
    """Remove <think>...</think> block — we parse only the final answer."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    cleaned = re.sub(r"<\|im_end\|>", "", cleaned)
    return cleaned.strip()


def parse_anomaly_detection(full_text):
    answer = _strip_think(full_text).lower()
    if not answer:
        return "Cannot Find"
    # Look for explicit yes/no patterns in the answer
    if re.search(r"\b(no\s+anomaly|not\s+detected|no\b)", answer):
        return "no"
    if re.search(r"\b(yes|anomaly\s+detected|detected|present)\b", answer):
        return "yes"
    if "yes" in answer and "no" not in answer:
        return "yes"
    if "no" in answer and "yes" not in answer:
        return "no"
    return "Cannot Find"


def parse_anomaly_bounds(full_text):
    answer = _strip_think(full_text)
    # Try explicit "starts at X and ends at Y" pattern first
    m = re.search(r"starts?\s+at\s+(\d+).*?ends?\s+at\s+(\d+)", answer, re.IGNORECASE)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    # Try "X, Y" or "X to Y" pattern
    m = re.search(r"\[?(\d+)\s*[,\-]\s*(\d+)\]?", answer)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    m = re.search(r"(\d+)\s+to\s+(\d+)", answer, re.IGNORECASE)
    if m:
        return [int(m.group(1)), int(m.group(2))]
    # Fallback: first two numbers in the answer
    nums = re.findall(r"\d+", answer)
    if len(nums) >= 2:
        return [int(nums[0]), int(nums[1])]
    return [0, 0]


def parse_root_cause(text):
    answer = _strip_think(text)
    cause = None
    for anomaly in anomaly_list:
        if anomaly.lower() in answer.lower():
            cause = anomaly
            break
    return cause

def parse_anomaly_length(text: str) -> int | None:
    match = re.search(r"<conclusion>(.*?)</conclusion>", text, re.DOTALL)
    if not match:
        match = re.search(r"\b\d+\b", text)
        if match:
            first_number = int(match.group(0))
        else:
            first_number = None
        return first_number

    block = match.group(1)

    length_match = re.search(r"Anomaly Length:\s*\[?(\d+)\]?", block)
    if length_match:
        return int(length_match.group(1))

    fallback_match = re.search(r"\b\d+\b", block)
    if fallback_match:
        return int(fallback_match.group(0))

    return None

def convert_intervals_to_binary(intervals, time_index):
    binary = np.zeros(len(time_index), dtype=int)
    for start, end in intervals:
        start = pd.to_datetime(start)
        end = pd.to_datetime(end)
        mask = (time_index >= start) & (time_index <= end)
        binary[mask] = 1
    return binary
