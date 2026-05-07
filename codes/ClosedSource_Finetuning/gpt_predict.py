import copy
import datetime
import json
import os
from openai import OpenAI
from pathlib import Path

client = OpenAI()


def predict_response_without_data(using_model, question, get_prompt=False):
    system_prompt = "Please predict the response to the given message"
    
    if get_prompt:
        return system_prompt, question
    
    response = client.chat.completions.create(
    model=using_model,
    messages=[
        {
            "role": "system",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": question
        }
    ]
    )

    return response.choices[0].message.content

def predict_json(base_model, using_model, input_path, lang):
    print("Processing file:", input_path)

    filename = input_path.stem

    # ディレクトリがなければ作成
    output_dir = Path("response") / base_model.replace("/", "_") / lang
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename}.json"

    if os.path.exists(output_path):
        print(f"Skipped: {output_path}")
        return


    #* チャットのデータ
    with open(input_path, "r") as f:
        all_day = json.load(f)
        
        #! 入力ファイルに追記して保存する
        output_json = copy.deepcopy(all_day)

        try:
            day5 = all_day["day_5"]
        except Exception:
            print(f"{filename} is missing day5 data")
            return


    for interaction in day5:
        question = interaction["message"]
        predicted_answer = predict_response_without_data(using_model, question)
        predicted_answer = predicted_answer.strip()
        #* 結果の書き込み
        output_json["day_5"][output_json["day_5"].index(interaction)]["prediction"] = predicted_answer
    
    #* promptも保存
    output_json["metadata"] = [{
        "model": using_model,
        "method": "Fine-tuning (GPT)",
        "predicted_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": predict_response_without_data(using_model, "question", get_prompt=True)[0],
        "user_prompt": predict_response_without_data(using_model, "question", get_prompt=True)[1],
        }]


    # JSONファイルとして保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
    print("Saved to:", output_path)

def main():
    # gpt-3.5-turbo, gpt-4o-mini-2024-07-18
    base_model = "gpt-4o-mini-2024-07-18"
    langs = ["jp", "en", "cn"]

    data_dir = Path("../../100人規模実験/anonymous_100")
    json_files = []
    for lang in langs:
        json_files.extend(list((data_dir/lang).glob("*.json")))

    model_name_path = Path(f"./model_name_{base_model}.json")
    with model_name_path.open("r", encoding="utf-8") as f:
        model_name_dict = json.load(f)

    for json_file in json_files:
        using_model = model_name_dict[json_file.stem]
        predict_json(base_model, using_model, json_file, json_file.stem[:2])


if __name__ == "__main__":
    main()