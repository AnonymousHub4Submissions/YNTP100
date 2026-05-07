# main.py
import os
import json
import time
import datetime
import copy
from pathlib import Path

# from llm.openai import OpenAIModel
from llm.deepseek import DeepSeekModel
# from llm.qwen import QwenModel
# from llm.gemini import GeminiModel
from predictor import CoTPredictor

def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(path: Path, data: dict):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

start_time = time.time()

lang = "jp"
method = "few-shot"   # "zero-shot" or "few-shot"
# using_model = "gpt-4o-mini"
using_model = "DeepSeek-R1-Distill-Qwen-14B"
# using_model = "Qwen3-14B"
# using_model = "gemini-2.5-flash"

json_dir = Path("../../100人規模実験/anonymous_100") / lang
json_files = list(json_dir.glob("*.json"))

output_root = Path("prompt_response") / method / using_model / lang
output_root.mkdir(parents=True, exist_ok=True)

# llm = OpenAIModel(using_model)
llm = DeepSeekModel(using_model)
# llm = QwenModel(using_model)
# llm = GeminiModel(using_model)
predictor = CoTPredictor(llm)

for json_file in json_files:
    print(f"start: {json_file}")
    output_path = output_root / f"{json_file.stem}.json"
    if output_path.exists():
        continue

    all_day = load_json(json_file)
    output_json = copy.deepcopy(all_day)

    train_text = None
    style = None
    if method == "few-shot":
        examples = []
        for day in range(1, 5):
            for inter in all_day[f"day_{day}"]:
                examples.append(
                    f"Q({inter['speaker']}): {inter['message']}\nA: {inter['response']}"
                )
        train_text =  "\n\n".join(examples)
        style = predictor.infer_style(train_text)

    print(f"style is \n{style}\n")
    for idx, inter in enumerate(all_day.get("day_5", [])):
        predicted = predictor.predict(
            question=inter["message"],
            train_data=train_text if method == "few-shot" else None,
            style=style
        )

        key = (
            "predicted_response_with_data"
            if method == "few-shot"
            else "predicted_response_without_data"
        )
        output_json["day_5"][idx][key] = predicted.strip()

    output_json["metadata"] = {
        "model": using_model,
        "method": method,
        "cot_policy": "latent_style_decomposition",
        "predicted_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "style": style,
    }

    save_json(output_path, output_json)

print(f"Total time: {time.time() - start_time:.2f}s")
