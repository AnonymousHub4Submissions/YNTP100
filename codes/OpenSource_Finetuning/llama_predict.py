from unsloth import FastLanguageModel
import json
import copy
import os
from pathlib import Path


max_seq_length = 1024
dtype = None
load_in_4bit = False


def predict_json(model_name, adapter_dir, original_json_path, lang):
    with open(original_json_path, 'r', encoding='utf-8') as f:
        interactions = json.load(f)
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name,
        max_seq_length = max_seq_length,
        dtype = dtype,
        load_in_4bit = load_in_4bit,
    )

    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

    ### Instruction:
    {}
    ### Input:
    {}
    ### Response:
    {}"""
    
    instruction = "Please predict the response to the given message"

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )
    # 推論モードに切り替え
    FastLanguageModel.for_inference(model)

    results = copy.deepcopy(interactions)
    
    if adapter_dir is not None:
        model.load_adapter(adapter_dir, adapter_name="default")

    for interaction in interactions["day_5"]:
        inputs = tokenizer(
            [
                alpaca_prompt.format(
                    instruction, # instruction
                    interaction["message"], # input
                    "", # output
                )
            ], return_tensors = "pt").to("cuda")

        from transformers import TextStreamer
        text_streamer = TextStreamer(tokenizer)
        _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 256)

        #* テキストを取り出す
        output = tokenizer.decode(_[0])
        output = output.split("### Response:")[-1].strip()
        output = output.replace(tokenizer.eos_token, '').strip()
        print("\n\n### Output:\n")
        print(output)
        
        #* resultsの該当箇所にpredicted_responseを追加する
        results["day_5"][results["day_5"].index(interaction)]["prediction"] = output


    results["metadata"] = {
        "model_name": model_name,
        "adapter_dir": adapter_dir,
        "alpaca_prompt": alpaca_prompt,
        "instruction": instruction,
        "original_json_path": original_json_path,
    }
    
    output_dir = Path("response") / model_name / lang
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(original_json_path).stem.replace('.', '_')
    output_path = output_dir / f"{filename}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


