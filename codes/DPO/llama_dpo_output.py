import os
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
from transformers import TextStreamer
from transformers import AutoTokenizer
import json
from pathlib import Path
import copy
import datetime


def _latest_checkpoint_dir(base_dir: Path) -> Path:
    """Return the latest numeric checkpoint-* subdirectory under base_dir.
    Ignores files and folders that don't match 'checkpoint-<int>'.
    Raises FileNotFoundError if none found.
    """
    candidates = []
    if not base_dir.exists():
        raise FileNotFoundError(f"Adapter base directory not found: {base_dir}")
    for name in os.listdir(base_dir):
        full = base_dir / name
        if not full.is_dir():
            continue
        if not name.startswith("checkpoint-"):
            continue
        suffix = name.split("-")[-1]
        try:
            step = int(suffix)
        except ValueError:
            continue
        candidates.append((step, full))
    if not candidates:
        raise FileNotFoundError(f"No numeric checkpoint-* directories found in {base_dir}")
    candidates.sort(key=lambda x: x[0])
    return candidates[-1][1]



def predict_json(original_json_path, model_name, adapter_path, lang):
    
    # 基本設定
    max_seq_length = 1024
    dtype = None
    load_in_4bit = False
    random_seed = 3407

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
        device_map="auto",  # 追加
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha = 16,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        use_gradient_checkpointing=False,
        random_state = random_seed,
        max_seq_length = max_seq_length,
        use_rslora=True,
    )
    # 推論モードに切り替え
    FastLanguageModel.for_inference(model)
    
    if adapter_path is not None:
        model.load_adapter(adapter_path, adapter_name="default")
    else:
        print("adapter_path is None. Skip loading adapter.")


    system_template = """
    <|start_header_id|>system<|end_header_id|>
    {system}
    <|eot_id|>
    """
    user_template = """
    <|start_header_id|>user<|end_header_id|>
    {user}
    <|eot_id|>
    """
    assistant_template = """
    <|start_header_id|>assistant<|end_header_id|>
    {assistant}
    <|eot_id|>
    """

    assistant_prefix = """
<|start_header_id|>assistant<|end_header_id|>
"""

    input_list = []

    with open(original_json_path, 'r', encoding='utf-8') as f:
        interactions = json.load(f)
    for interaction in interactions["day_5"]:
        input_list.append(interaction["message"])
    
    output_json = copy.deepcopy(interactions)
    
    
    for interaction in input_list:
        instruction = "You are a resident of a shared house. Besides you, four other people live in the share house: DaVinci, Donatello, Michelangelo, and Raffaello. Please think of a response to the conversation."
        # prompt = system_template.format(system=instruction)
        # prompt += user_template.format(user=interaction)
        # prompt += assistant_prefix
        prompt = system_template.format(system=instruction).strip() + "\n" + user_template.format(user=interaction).strip() + "\n" + assistant_prefix.strip()
        inputs = tokenizer([prompt], return_tensors="pt").to("cuda")

        generated_output = model.generate(**inputs, max_new_tokens=1024)
        output_text = tokenizer.decode(generated_output[0], skip_special_tokens=True)
        # print("=== Generated Output ===")
        # print(output_text)
        # output_textのうち、prompt部分を取り除く
        try:
            output_text = output_text.split(interaction.strip())[-1].strip()
        except:
            pass
        # if "assistant\n\n" in output_text:
        # assistantが先頭にあるときのみ除去するように修正
        if output_text.startswith("assistant"):
            output_text = output_text[len("assistant"):].strip()
        print(output_text)
        output_json["day_5"][input_list.index(interaction)]["prediction"] = output_text
        print("\n" + "="*40 + "\n")
    
    output_json["metadata"] = {
        "model": model_name,
        "instruction": instruction,
        "chat_template": system_template + user_template + assistant_template,
        "method": "DPO with PEFT",
        "predicted_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
    
    output_dir = Path("response") / model_name / lang
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(original_json_path).stem.replace('.', '_')
    output_path = output_dir / f"{filename}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)

