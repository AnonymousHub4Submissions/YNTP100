import unsloth
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from unsloth import FastLanguageModel
from transformers import TextStreamer
import copy
import os
import json
import datetime
from pathlib import Path



def predict_json(original_json_path, model_name, adapter_path, lang,):
    
    original_tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )

    original_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    
    input_list = []
    with open(original_json_path, 'r', encoding='utf-8') as f:
        interactions = json.load(f)
    for interaction in interactions["day_5"]:
        input_list.append(interaction["message"])
    
    output_json = copy.deepcopy(interactions)
    

    print("set use_Prompt_Engineering = False")

    #* 学習済みモデル
    aligned_tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=True)
    aligned_model = AutoModelForCausalLM.from_pretrained(
        adapter_path,
        torch_dtype=torch.bfloat16,         # save_16bit を使ったため
        device_map="auto",
        trust_remote_code=True,
    )
    
    system_prompt = f"""You are a resident who has just moved into the share house. Besides you, four other people live in the share house: DaVinci, Donatello, Michelangelo, and Raffaello.Please predict responses and output only the answer part."""

    for interaction in input_list:
        question = interaction
        inputs = original_tokenizer(
            [
                original_tokenizer.apply_chat_template(
                    [
                        {"role" : "system", "content" : system_prompt},
                        {"role" : "user", "content" : f"{question}"},
                        # {"role" : "user", "content" : f"Q. {question}"},
                    ],
                    tokenize = False,
                    add_generation_prompt = True,
                )
            ], return_tensors = "pt").to("cuda")

        
        # streamerではなく直接generate
        generated_ids = original_model.generate(
            **inputs,
            max_new_tokens = 1024,
            temperature = 0.7, top_p = 0.8, top_k = 20
        )
        generated_text = original_tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        if "assistant" in generated_text:
            try:
                #* assistant以降を抽出
                generated_text = generated_text.split("assistant")[-1].strip()
            except :
                continue
        if "A." in generated_text:
            try:
                #* A.以降を抽出
                generated_text = generated_text.split("A.")[-1].strip()
            except :
                continue
        print("question:", question)
        print("generated_text before alignment:", generated_text)
        print()

        #* alignerを通して修正
        aligner_user_prompt = "BEGINNING OF CONVERSATION: USER: Edit the following Question-Answer pair to make it sound more like a human's statement: {question} | {answer} ASSISTANT:"
        input = aligner_user_prompt.format(
            question=question,
            answer=generated_text,
            )

        input_ids = aligned_tokenizer.encode(input, return_tensors='pt').cuda()
        output_ids = aligned_model.generate(input_ids, max_new_tokens=1024)[0]
        output_text = aligned_tokenizer.decode(output_ids, skip_special_tokens=True)
        print(output_text)
        # 最初のASSISTANT:以降を抽出, ASSISTANT:が複数あれば先頭で切る
        try:
            output_text = output_text.split("ASSISTANT:")[1].strip()
        except:
            pass
        
        output_json["day_5"][input_list.index(interaction)]["prediction"] = output_text
        print("\n" + "="*40 + "\n")
        
    output_json["metadata"] = {
        "model": model_name,
        "method": "aligner",
        "instruction": system_prompt,
        "predicted_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    
    # 保存 dataset/response/modelname/lang/ jp_1.json, etc.
    output_dir = os.path.join("response", model_name.replace(".", "_").replace("/", "_"), lang)
    os.makedirs(output_dir, exist_ok=True)
    # output_path = os.path.join(output_dir, f"{adapter_path.split('/')[-1]}.json")
    # adapterのフォルダ名と同じファイル名で保存
    output_path = os.path.join(output_dir, f"{Path(original_json_path).stem}.json")
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)



