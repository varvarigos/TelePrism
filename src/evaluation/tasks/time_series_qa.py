import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
# from teleprism.utils.api_call import send_to_llm
import re
import os

def ground_truth_stats(no_anomaly_seq_list):
    means_by_channel = []
    variances_by_channel = []

    channels = [ch for ch in no_anomaly_seq_list[0]['dataframe'].keys() if ch not in ["UL_Protocol", "DL_Protocol", "timestamp", "Jamming"]]
    trend_by_channel = {ch: [] for ch in no_anomaly_seq_list[0]['dataframe'].keys() if ch not in ["UL_Protocol", "DL_Protocol", "timestamp", "Jamming"]}
    per_by_channel = {ch: [] for ch in no_anomaly_seq_list[0]['dataframe'].keys() if ch not in ["UL_Protocol", "DL_Protocol", "timestamp", "Jamming"]}

    for entry in no_anomaly_seq_list:
        df = pd.DataFrame(entry['dataframe']).drop(columns=["timestamp", "UL_Protocol", "DL_Protocol", "Jamming"], errors='ignore')
        
        #Parse None values
        df.replace("None", np.nan, inplace=True)

        #Convert string to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        means_by_channel.append(df.mean())
        variances_by_channel.append(df.var())

        x = np.arange(len(df))

        for channel in df.columns:
            y = df[channel].values

            if np.all(np.isnan(y)):
                trend_by_channel[channel].append(np.nan)
                per_by_channel[channel].append(np.nan)
                continue

            slope, _ = np.polyfit(x, y, 1)
            trend_by_channel[channel].append(slope)
            periodicity = periodicity_calculation(y)
            per_by_channel[channel].append(periodicity)

    #Convert to dataframes
    means_df = pd.DataFrame(means_by_channel, columns=channels)
    variances_df = pd.DataFrame(variances_by_channel, columns=channels)
    trend_df = pd.DataFrame(trend_by_channel)
    per_df = pd.DataFrame(per_by_channel)

    mean_slopes_by_channel = trend_df.mean()
    std_slopes_by_channel = trend_df.std()
    for channel in df.columns:
        mean_slope=mean_slopes_by_channel[channel]
        std_slope=std_slopes_by_channel[channel]
        trend_df[channel] = trend_df[channel].apply(lambda x: 1 if x > mean_slope + std_slope else (-1 if x < mean_slope - std_slope else 0))

    return means_df, variances_df, trend_df, per_df, mean_slopes_by_channel, std_slopes_by_channel

def llm_generated_stats(QA_eval_list, eval_dict, trend_df, provider=None, model=None, tokenizer=None, device=None, dtype=None):
    #Ask ChatGPT
    if hasattr(model, "ts_encoder"):
        multimodal = True
    else:
        multimodal = False
        
    mean, var, trend, per = eval_dict["mean"], eval_dict["var"], eval_dict["trend"], eval_dict["per"]
    
    num_series = len(QA_eval_list)
    keys = [ch for ch in QA_eval_list[0]['dataframe'].keys() if ch not in ["timestamp", "UL_Protocol", "DL_Protocol", "Jamming"]]
    gpt_mean_by_channel = {ch: [np.nan]* num_series  for ch in keys}
    gpt_var_by_channel = {ch: [np.nan]* num_series for ch in keys}
    gpt_trend_by_channel = {ch: [np.nan]* num_series  for ch in keys}
    gpt_per_by_channel = {ch: [np.nan]* num_series  for ch in keys}
    #for each series(dataframe), and each channel, give chatgpt the series and ask the following quesiotns
    #extract responses and store in dataframe 

    trend_dict = {
        (channel, i): 0
        for channel in keys
        for i in (-1, 0, 1)
    }

    print("QA_EVAL LIST", len(QA_eval_list))
    for i, entry in enumerate(QA_eval_list):
        df = pd.DataFrame(entry['dataframe']).drop(columns=["timestamp", "UL_Protocol", "DL_Protocol", "Jamming"], errors='ignore')
        #Parse None values
        df.replace("None", np.nan, inplace=True)

        #Convert string to numeric
        df = df.apply(pd.to_numeric, errors='coerce')
        
        if multimodal:
            time_series = entry["ts_array"]
        else:
            time_series = None

        for channel in df.columns:
            if multimodal:
                y = ""
            else:
                y = df[channel].values
            
            if trend and trend_dict[(channel, int(trend_df[channel][i]))] != 5:
                trend_message = (f"Consider the following series: {y.tolist()}. "
                f"Please describe the average trend of the series, ignoring any NaN values. "
                f"If the series is decreasing on average, respond with a value of -1."
                "If it is increasing, respond with a value of 1. If there doesn't appear "
                "to be a strong trend in any direction, please respond with a value of 0. Note that wireless data can be noisy, so look at global changes to determine trend. "
                "Do not include any other numbers in your response whether in the form of "
                "intermediate calculations or steps. ONLY RESPOND WITH -1, 0, or 1. Please DO NOT include any other analysis or explanations. ")
                try:
                    response = send_to_llm(trend_message, provider, "time_series_qa", time_series, model, tokenizer, device, dtype)
                    response = extract_number(response)
                except:
                    response = np.nan
                gpt_trend_by_channel[channel][i] = response
                trend_dict[(channel, int(trend_df[channel][i]))] += 1

            if mean:
                y = df[channel].values
                message = (f"Consider the following list of numbers representing a time series: {y.tolist()}. "
                f"Some values may be missing (NaN). What is the average {channel} value of this series, "
                f"ignoring NaNs? Respond with only a single float rounded to 2 decimal places — no other text or numbers. Please DO NOT include any other analysis or explanations. "
                )
            # try:
                print(message)
                response = send_to_llm(message, provider, "time_series_qa", time_series, model, tokenizer, device=None, dtype=None)
                response = extract_number(response)
            # except:
            #     response = np.nan
                gpt_mean_by_channel[channel][i] = response

            if var:
                message = (f"Consider the following list of numbers representing a time series: {y.tolist()}. "
                f"Some values may be missing (NaN). What is the variance of {channel} for this series, "
                f"ignoring NaNs? Respond with only a single float rounded to 2 decimal places — no other text or numbers. Please DO NOT include any other analysis or explanations. "
                )
                try:
                    response = send_to_llm(message, provider, "time_series_qa", time_series, model, tokenizer, device, dtype)
                    response = extract_number(response)
                except:
                    response = np.nan
                gpt_var_by_channel[channel][i] = response

            if per:
                seq_len = len(df)
                periodicity_message = (f"Consider the following series: {y.tolist()}. Please "
                f"investigate whether the series exhibits strong periodicity, ignoring any NaN values. If it does, please"
                f"respond with with an integer value representing approximately how often strong"
                f"periods occur in the series. If there is no evidence of strong periodicity please"
                f"respond with the sequence length {seq_len}. Do not include any other numbers in your response whether"
                f"in the form of intermediate calculations or steps. Remember you MUST return an INTEGER value or {seq_len}. Please DO NOT include any other analysis or explanations. ")
                try:
                    response = send_to_llm(periodicity_message, provider, "time_series_qa", time_series, model, tokenizer, device, dtype)
                    response = extract_number(response)
                except:
                    response = np.nan
                gpt_per_by_channel[channel][i] = response

    gpt_mean_df = pd.DataFrame(gpt_mean_by_channel)
    gpt_var_df = pd.DataFrame(gpt_var_by_channel)
    gpt_trend_df = pd.DataFrame(gpt_trend_by_channel)
    gpt_per_df = pd.DataFrame(gpt_per_by_channel)
    return gpt_mean_df, gpt_var_df, gpt_trend_df, gpt_per_df

def time_series_qa_eval(no_anomaly_seq_list, eval_dict, results_dir, provider=None, model=None, tokenizer=None, device=None, dtype=None):
    means_df, variances_df, trend_df, per_df, _, _ = ground_truth_stats(no_anomaly_seq_list)
    gpt_mean_df, gpt_var_df, gpt_trend_df, gpt_per_df = llm_generated_stats(no_anomaly_seq_list, eval_dict, trend_df, provider, model, tokenizer, device, dtype)
    # means_df = means_df[:num_series_analyze]
    # variances_df = variances_df[:num_series_analyze]
    # trend_df = trend_df[:num_series_analyze]
    # per_df = per_df[:num_series_analyze]

    squared_error_df_means = (gpt_mean_df - means_df) ** 2
    squared_error_df_var = (gpt_var_df - variances_df) ** 2
    squared_error_df_per = (gpt_per_df - per_df) ** 2

    mse_by_channel_means = squared_error_df_means.mean()
    mse_by_channel_var = squared_error_df_var.mean()
    mse_by_channel_per = squared_error_df_per.mean()

    abs_error_df_means = abs(gpt_mean_df - means_df) 
    abs_error_df_var = abs(gpt_var_df - variances_df) 
    abs_error_df_per = abs(gpt_per_df - per_df) 

    mae_by_channel_means = abs_error_df_means.mean()
    mae_by_channel_var = abs_error_df_var.mean()
    mae_by_channel_per = abs_error_df_per.mean()

    #Accuracy for Trend
    #checking for NaNs
    conditional_accuracies = {}
    for label in [-1, 0, 1]:
        mask = (trend_df == label)
        correct = (gpt_trend_df == trend_df) & mask
        conditional_accuracy = correct.sum() / mask.sum()
        conditional_accuracies[label] = conditional_accuracy      

    conditional_accuracy_df = pd.DataFrame(conditional_accuracies)
    conditional_accuracy_df.columns = ['True Decreasing (-1)', 'True Stable (0)', 'True Increasing(1)']

    if eval_dict["trend"]: 
        conditional_accuracy_df.to_csv(f"{results_dir}/periodicity_accuracy.csv", index=True)
    if eval_dict["mean"]:
        mse_by_channel_means.to_csv(f"{results_dir}/mse_means.csv", index=True)
        mae_by_channel_means.to_csv(f"{results_dir}/mae_means.csv", index=True)
    if eval_dict["var"]:
        mse_by_channel_var.to_csv(f"{results_dir}/mse_var.csv", index=True)
        mae_by_channel_var.to_csv(f"{results_dir}/mae_var.csv", index=True)
    if eval_dict["per"]:
        mse_by_channel_per.to_csv(f"{results_dir}/mse_periodicity.csv", index=True)
        mae_by_channel_per.to_csv(f"{results_dir}/mae_periodicity.csv", index=True)
        

    return {
        "ground_truth": {
            "mean": means_df, 
            "var": variances_df, 
            "trend": trend_df,
            "per": per_df
            }, 
        "predictions": {
            "mean": gpt_mean_df, 
            "var": gpt_var_df, 
            "trend": gpt_trend_df,
            "per": gpt_per_df
            }
        }

#Periodicity calculation
def periodicity_calculation(y):
    yf = np.abs(rfft(y - y.mean())) #generates frequencies of ith component of yf
    xf = rfftfreq(len(y), d=1)  # d=1 assuming unit sample spacing, find dominant frequency index, skipping the zero component (representing the mean), adding 1 since we skipped the first component
    peak_freq_idx = np.argmax(yf[1:]) + 1  # ignore the zero-frequency component, using this index, find the dominant period
    dominant_freq = xf[peak_freq_idx] #TODO Look into periodicity distribution 
    if dominant_freq > 0:
        period = round(1 / dominant_freq, 2)
        periodicity = period #strong periodicity detected every period periods
    else:
        periodicity = 0
    return periodicity

def extract_number(response):
    # Strip <think>...</think> first — parse only the final answer.
    answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    answer = re.sub(r"<\|im_end\|>", "", answer).strip()
    if not answer:
        answer = response
    match = re.search(r"[-+]?(?:\d+\.\d+|\d+)", answer)
    return np.float64(match.group()) if match else None