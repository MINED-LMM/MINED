import os
import sys
import json
import re
import string
from collections import defaultdict, Counter


INPUT_TEMPLATE = (
    "./final_output/{task}/output_from_{model}.jsonl"
)


TASKS = [
    # "time_agnostic",
    # "timestamp",
    # "awareness_future",
    # "awareness_past",
    # "future_unanswerable_date",
    # "previous_unanswerable_date",
    # "ranking",
    # "understanding",
    # "calculation",
# "temporal_interval",
    "robustness",
]


MODELS = [ "Qwen2.5-VL-7B-Instruct"]




TOOLS_DIRS = [
    "eval_code",  
]
# ============================================================


PRED_FIELDS = [
    "image_mm_questions_prediction",
    "paraphrase_image_mm_questions_prediction",
    "image_mm_paraphrase_completion_prediction",
    "paraphrase_image_mm_paraphrase_completion_prediction",
]


def normalize_answer(text: str) -> str:
    def remove_articles(s: str) -> str:
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", s)

    def white_space_fix(s: str) -> str:
        return " ".join(s.split())

    def remove_punc(s: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in s if ch not in exclude)

    def lower(s: str) -> str:
        return (s or "").lower()

    text = lower(text)
    text = remove_punc(text)
    text = remove_articles(text)
    text = white_space_fix(text)
    return text


def get_tokens(text: str):
    if not text:
        return []
    return normalize_answer(text).split()


def compute_f1_score(predicted_text: str, gold_text: str) -> float:
    gold_tokens = get_tokens(gold_text)
    predicted_tokens = get_tokens(predicted_text)
    if len(gold_tokens) == 0 or len(predicted_tokens) == 0:
        return float(gold_tokens == predicted_tokens)
    common = Counter(gold_tokens) & Counter(predicted_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(predicted_tokens)
    recall = num_same / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def compute_cem_score(predicted_text: str, gold_text: str) -> float:
    """
    Compute CEM score: check whether gold_text fully appears in predicted_text.
    Return 1.0 if gold_text is a substring of predicted_text after normalization, else 0.0.
    """
    if not gold_text or not predicted_text:
        return 0.0
    
    # Normalize inputs
    normalized_gold = normalize_answer(gold_text)
    normalized_pred = normalize_answer(predicted_text)
    
    # Check substring containment
    if normalized_gold in normalized_pred:
        return 1.0
    else:
        return 0.0


def build_cem_evaluator():
    # Prefer using in-project VQAEval if available
    for tools_dir in TOOLS_DIRS:
        try:
            if tools_dir and tools_dir not in sys.path:
                sys.path.insert(0, tools_dir)
            from tools import VQAEval  # type: ignore

            evaluator = VQAEval()

            def cem_with_vqaeval(pred: str, gold: str) -> float:
                return float(evaluator.evaluate(str(pred or ""), [str(gold or "")]))

            return cem_with_vqaeval
        except Exception:
            continue

    # Fallback: use custom CEM computation
    return compute_cem_score


def read_jsonl(file_path: str):
    rows = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(file_path: str, rows) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def write_txt(file_path: str, content: str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


def evaluate_entry(entry: dict, compute_cem) -> tuple[float, float, int, list, list]:
    gold_answer = entry.get("current_answer", "")
    cem_values = []
    f1_values = []
    cem_list = []
    f1_list = []

    for field_name in PRED_FIELDS:
        predicted_text = entry.get(field_name, "")
        
        # Check if the field is non-empty
        if isinstance(predicted_text, str) and predicted_text.strip() != "":
            cem_score = compute_cem(predicted_text, gold_answer)
            f1_score = compute_f1_score(predicted_text, gold_answer)
            
            cem_values.append(cem_score)
            f1_values.append(f1_score)
            cem_list.append(cem_score)
            f1_list.append(f1_score)
        else:
            # Field is empty, record as null
            cem_list.append(None)
            f1_list.append(None)

    if not cem_values:
        return 0.0, 0.0, 0, cem_list, f1_list

    return sum(cem_values) / len(cem_values), sum(f1_values) / len(f1_values), len(cem_values), cem_list, f1_list


def process_one_file(input_path: str, compute_cem) -> tuple[str, str]:
    items = read_jsonl(input_path)

    updated_items = []
    total_cem = 0.0
    total_f1 = 0.0
    num_items = 0
    type_to_items = defaultdict(list)

    for item in items:
        cem_avg_raw, f1_avg_raw, _, cem_list, f1_list = evaluate_entry(item, compute_cem)

        # Scale CEM/F1 to percentage
        cem_avg = cem_avg_raw * 100.0
        f1_avg = f1_avg_raw * 100.0

        item["cem"] = cem_avg
        item["f1"] = f1_avg
        item["cem_list"] = cem_list
        item["f1_list"] = f1_list
        updated_items.append(item)

        total_cem += cem_avg
        total_f1 += f1_avg
        num_items += 1

        item_type = item.get("type", "unknown")
        type_to_items[item_type].append(item)

    overall_cem = (total_cem / num_items) if num_items > 0 else 0.0
    overall_f1 = (total_f1 / num_items) if num_items > 0 else 0.0

    base_dir = os.path.dirname(input_path)
    base_name = os.path.basename(input_path)

    # Output evaluated JSONL: eval_<original filename>
    out_jsonl = os.path.join(base_dir, f"eval_{base_name}")
    write_jsonl(out_jsonl, updated_items)

    # Output summary TXT: same name with .txt extension
    base_no_ext, _ = os.path.splitext(base_name)
    out_txt = os.path.join(base_dir, f"{base_no_ext}.txt")

    lines = []
    lines.append(f"Total: {num_items}")
    lines.append(f"Overall average: CEM={overall_cem:.4f}, F1={overall_f1:.4f}")
    lines.append("")
    lines.append(f"Total number of types: {len(type_to_items)}")
    lines.append("Type stats (count desc):")
    for t, group in sorted(type_to_items.items(), key=lambda kv: len(kv[1]), reverse=True):
        if group:
            t_cem = sum(x.get("cem", 0.0) for x in group) / len(group)
            t_f1 = sum(x.get("f1", 0.0) for x in group) / len(group)
        else:
            t_cem = 0.0
            t_f1 = 0.0
        lines.append(f"{t} {len(group)} {t_cem:.4f} {t_f1:.4f}")

    write_txt(out_txt, "\n".join(lines))

    print(f"[OK] {base_name} => JSONL: {out_jsonl} | TXT: {out_txt}")
    print(f"    Overall: CEM={overall_cem:.4f}, F1={overall_f1:.4f}")
    return out_jsonl, out_txt


def main():
    if not MODELS:
        print("Please fill the MODELS list at the top of the file before running.")
        return
    if not TASKS:
        print("Please fill the TASKS list at the top of the file before running.")
        return

    compute_cem = build_cem_evaluator()

    for task_name in TASKS:
        for model_name in MODELS:
            input_path = INPUT_TEMPLATE.format(task=task_name, model=model_name)
            if not os.path.exists(input_path):
                print(f"[SKIP] Input not found: {input_path}")
                continue
            try:
                process_one_file(input_path, compute_cem)
            except Exception as exc:
                print(f"[ERROR] Failed to process: {input_path}, reason: {exc}")


if __name__ == "__main__":
    main()