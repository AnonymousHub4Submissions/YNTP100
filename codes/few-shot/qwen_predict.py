from unsloth import FastLanguageModel
import json
import copy
import os
from pathlib import Path


max_seq_length = 1024
dtype = None
load_in_4bit = False


def predict_json(original_json_path, model_name, lang):
    with open(original_json_path, 'r', encoding='utf-8') as f:
        interactions = json.load(f)
        
        train_day_list = []
        for i in range(1, 5):
            # day_data = all_day["daily_interactions"][f"day_{i}"]
            day_data = interactions[f"day_{i}"]
            for interaction in day_data:
                train_day_list.append({
                    "question": interaction["message"],
                    "answer": interaction["response"],
                })
        print("Train day list:", train_day_list)
        
    #* Q. auestion, A. answerの形式のexampleテキストを作る
    messages_four = train_day_list[:4]
    QA_examples = ""
    for msg in messages_four:
        QA_examples += f"Q. {msg['question']}\nA. {msg['answer']}\n\n"
    
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
    
    instruction = f"""You are a resident who has just moved into the share house. Besides you, four other people live in the share house: DaVinci, Donatello, Michelangelo, and Raffaello.Please predict responses referencing below Q&A examples and output only the answer part: \n{QA_examples}"""

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

    for interaction in interactions["day_5"]:
        # inputs = tokenizer(
        #     [
        #         alpaca_prompt.format(
        #             instruction, # instruction
        #             interaction["message"], # input
        #             "", # output
        #         )
        #     ], return_tensors = "pt").to("cuda")
        
        # from transformers import TextStreamer
        # text_streamer = TextStreamer(tokenizer)
        # _ = model.generate(**inputs, streamer = text_streamer, max_new_tokens = 256)

        # #* テキストを取り出す
        # output = tokenizer.decode(_[0])
        # output = output.split("### Response:")[-1].strip()
        # output = output.replace(tokenizer.eos_token, '').strip()
        # #* "###"の手前まで使う
        # output = output.split("###")[0].strip()
        # print("\n\n### Output:\n")
        # print(output)
        
        # #* resultsの該当箇所にpredicted_responseを追加する
        # results["day_5"][results["day_5"].index(interaction)]["prediction"] = output
        
        inputs = tokenizer(
        [
            tokenizer.apply_chat_template(
                [
                    {"role" : "system", "content" : instruction},
                    {"role" : "user", "content" : f"{interaction['message']}"},
                ],
                tokenize = False,
                add_generation_prompt = True,
                enable_thinking = False,
            )
        ], return_tensors = "pt").to("cuda")

        
        # streamerではなく直接generate
        generated_ids = model.generate(
            **inputs,
            max_new_tokens = 1024,
            temperature = 0.7, top_p = 0.8, top_k = 20
        )
        generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if "</think>\n\n" in generated_text:
            #* assistant以降を抽出
            generated_text = generated_text.split("</think>\n\n")[-1].strip()
        if "A." in generated_text:
            #* A.以降を抽出
            generated_text = generated_text.split("A.")[-1].strip()
        print("generated_text:", generated_text)
            
        #* resultsの該当箇所にpredicted_responseを追加する
        results["day_5"][results["day_5"].index(interaction)]["prediction"] = generated_text


    results["metadata"] = {
        "model_name": model_name,
        # "prompt": chat_template,
        "alpaca_prompt": alpaca_prompt,
        "instruction": instruction,
    }
    
    output_dir = Path("response") / model_name.replace("/", "_") / lang
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = Path(original_json_path).stem.replace('.', '_')
    output_path = output_dir / f"{filename}.json"
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


