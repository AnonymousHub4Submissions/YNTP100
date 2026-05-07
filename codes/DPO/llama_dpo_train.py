import os
from unsloth import is_bfloat16_supported
from unsloth import FastLanguageModel, PatchDPOTrainer
PatchDPOTrainer()
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset
import json
import math
from pathlib import Path



def DPO_train(model_name, train_dataset, num_train_epochs, lang):
    
    dpo_output = Path("adapter") / model_name.replace("/", "_") / lang / Path(train_dataset).stem.replace('.', '_')
    
    
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

    #! llama3.1の場合
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
    
    def apply_chat_template(example):
        prompt = system_template.format(system=system_prompt)
        prompt += user_template.format(user=example['prompt'])
        
        example["chosen"] = assistant_template.format(assistant=example['chosen'])
        example["rejected"] = assistant_template.format(assistant=example['rejected'])
        example["prompt"] = prompt
        return example
    train_dataset = load_dataset("json", data_files=train_dataset)
    dataset = train_dataset.map(
        apply_chat_template,
        num_proc = num_proc,
    )
    

    print(dataset['train'][0])

    # 訓練時に何ステップごとに保存するか計算する
    # per-device batch と gradient_accumulation を変数にして使いやすくする
    per_device_train_batch_size = 1
    gradient_accumulation_steps = 8
    dataset_size = len(dataset['train'])
    effective_batch = per_device_train_batch_size * gradient_accumulation_steps
    # ステップ数（最適化ステップ）/ エポック
    steps_per_epoch = math.ceil(dataset_size / effective_batch) if effective_batch > 0 else 0
    # 1エポックごとに保存 => 保存間隔(ステップ) = 1 * steps_per_epoch
    save_steps = max(1, 1 * steps_per_epoch)
    print(f"dataset_size={dataset_size}, effective_batch={effective_batch}, steps_per_epoch={steps_per_epoch}, save_steps={save_steps}")

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
        max_seq_length = max_seq_length,
        use_rslora=True,
    )

    trainer = DPOTrainer(
        model = model,
        ref_model = None,
        train_dataset = dataset['train'],
        tokenizer = tokenizer,
        beta=0.1,
        max_length = max_seq_length,
        max_prompt_length = 1024,
        device_map="auto",  # 追加
        args = DPOConfig(
            per_device_train_batch_size=per_device_train_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            num_train_epochs= num_train_epochs,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 10,
            save_strategy = "steps",
            save_steps = save_steps,
            save_total_limit = 2,
            output_dir = dpo_output,
            learning_rate = 5e-6,
            warmup_steps=10,
            lr_scheduler_type="linear",
            optim="adamw_8bit",
    ),
    )

    trainer.train()
