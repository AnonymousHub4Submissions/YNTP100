import json
import os
from pathlib import Path


def format_data(original_data):
    samples = []
    for i in range(1, 5):
        for interaction in original_data[f"day_{i}"]:
            samples.append({
                "messages": [
                    {"role": "system", "content": "Please predict the response to the given message"},
                    {"role": "user", "content": interaction["message"]},
                    {"role": "assistant", "content": interaction["response"]}
                ]
            })
    return samples

def main():
    langs = ["jp", "en", "cn"]
    data_dir = Path("../../100人規模実験/anonymous_100")
    save_dir = Path("./dataset")

    for lang in langs:
        os.makedirs(save_dir/lang, exist_ok=True)
        for file_name in os.listdir(data_dir/lang):
            if not file_name.endswith(".json"): continue

            with open(data_dir/lang/file_name, "r") as f:
                original_data = json.load(f)
            samples = format_data(original_data)

            save_name = os.path.basename(file_name).replace(".json", ".jsonl")

            with open(save_dir/lang/save_name, "w") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
