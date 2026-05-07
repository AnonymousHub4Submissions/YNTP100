import json
import os


def create_dataset(lang, json_path):
    with open(json_path, "r") as f:
        all_day = json.load(f)
    instruction = "Please predict the response to the given message"
    dataset = []
    for i in range(1, 5):
        for interaction in all_day[f"day_{i}"]:
            dataset.append({
                "instruction": instruction,
                "input": interaction["message"],
                "output": interaction["response"]
            })
    # 保存
    output_dir = f"dataset/{lang}"
    os.makedirs(output_dir, exist_ok=True)
    # ファイル名はtrain_data_<元のファイル名>
    base_name = os.path.basename(json_path).replace(".json", "")
    output_path = os.path.join(output_dir, f"train_data_{base_name}.json")
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

