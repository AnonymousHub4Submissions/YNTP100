import os
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import json
import math
from pathlib import Path
import copy


def create_dataset(model_name, original_json_path, lang):
    output_dir = Path("dataset") / model_name.replace("/", "_") / lang
    # output_path = Path("dataset") / model_name.replace("/", "_") / Path(original_json_path).stem.replace('.', '_')
    
    with open(original_json_path, 'r', encoding='utf-8') as f:
        all_day_data = json.load(f)
    
    empty_dataset = []
    for i in range(1, 5):
        day_data = all_day_data[f"day_{i}"]
        for interaction in day_data:
            prompt = interaction["message"]
            response = interaction["response"]
            empty_dataset.append({
                "prompt": prompt,
                "chosen": response,
                "rejected": ""
            })
    
    # 基本設定
    max_seq_length = 2048
    dtype = None
    load_in_4bit = False
    num_proc = 4
    sft_model = model_name

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = sft_model,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map="auto",  # 追加
    )

    system_prompt = "You are a resident of a shared house. Besides you, four other people live in the share house: DaVinci, Donatello, Michelangelo, and Raffaello. Please think of a response to the conversation. Please respond in the language used in the conversation."

    # Llama-3.1用のチャットテンプレートを設定
    if tokenizer.chat_template is None:
        chat_template = """{% if messages[0]['role'] == 'system' %}{% set system_message = messages[0]['content'] %}{% set messages = messages[1:] %}{% else %}{% set system_message = "" %}{% endif %}{% for message in messages %}{% if message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>

{{ message['content'] }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

{% elif message['role'] == 'assistant' %}{{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}"""
        tokenizer.chat_template = chat_template
    
    
    completed_dataset = copy.deepcopy(empty_dataset)
    
    for interaction in empty_dataset:
        # まずはrejectedを作成
        prompt = interaction["prompt"]
        inputs = tokenizer(
        [
            tokenizer.apply_chat_template(
                [
                    {"role" : "system", "content" : system_prompt},
                    {"role" : "user", "content" : prompt},
                ],
                tokenize = False,
                add_generation_prompt = True,
            )
        ], return_tensors = "pt").to("cuda")

        generated_ids = model.generate(
            **inputs,
            max_new_tokens = 1024,
            temperature = 0.7, top_p = 0.8, top_k = 20
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        try:
            generated_text = generated_text.split(prompt.strip())[-1].strip()
        except :
            continue
        print(generated_text)
        
        # rejectedにLLMの応答を入れる
        completed_dataset[completed_dataset.index(interaction)]["rejected"] = generated_text
    
    # データセットを保存
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(original_json_path).stem.replace('.', '_')}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(completed_dataset, f, ensure_ascii=False, indent=4)

create_dataset(
    model_name = "meta-llama/Llama-3.1-8B",
    original_json_path = "/home/data/kato/mymodel_train/mymodel-train/100人規模実験/anonymous_100/jp/jp_1.json",
    lang = "jp",
)

# create_dataset(
#     model_name = "meta-llama/Llama-3.1-8B-Instruct",
#     original_json_path = "/home/data/kato/mymodel_train/mymodel-train/100人規模実験/anonymous_100/jp/jp_1.json",
#     lang = "jp",
# )


# create_dataset(
#     model_name = "elyza/Llama-3-ELYZA-JP-8B",
#     original_json_path = "/home/data/kato/mymodel_train/mymodel-train/100人規模実験/anonymous_100/jp/jp_1.json",
#     lang = "jp",
# )

