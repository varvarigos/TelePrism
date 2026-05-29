import regex as re
from teleprism.dataset.qa_templates import QA_templates

def classify_answer(answer: str, answer_dict: dict, tag) -> str | None:
    answer = answer.lower()
    for label, pairs in answer_dict.items():
        for _, ans in pairs:
            if ans.lower() in answer:
                if tag == 'cong':
                    if 'not' in label:
                        return "No"
                    else:
                        return "Yes"
                elif tag =='motion':
                    dict_map = {
                        "Yes": "mobile",
                        "No": "static"
                        }
                    return dict_map[label]

                else:
                    return label
    return None

def parse_tagged_response(response: str, *, tag: str, options: list[str]) -> str:
    # Strip <think>...</think> first — parse only the final answer.
    answer = re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)
    answer = re.sub(r"<\|im_end\|>", "", answer).strip()
    if not answer:
        answer = response  # fallback if nothing left

    options_pattern = "|".join(re.escape(opt) for opt in options)
    tag_pattern = fr"<{re.escape(tag)}>\s*\[?({options_pattern})\]?\s*</{re.escape(tag)}>"

    match = re.search(tag_pattern, answer, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()

    ignore_case = any(len(opt) > 1 for opt in options)
    flags = re.IGNORECASE if ignore_case else 0

    fallback_pattern = r"\b(" + options_pattern + r")\b"
    fallback_match = re.search(fallback_pattern, answer, flags)

    if fallback_match:
        return fallback_match.group(1).capitalize()

    ans = classify_answer(answer, QA_templates[tag], tag)

    return ans