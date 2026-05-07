from unsloth import FastLanguageModel
import torch


    # import argparse

    # parser = argparse.ArgumentParser(description="Finetune Llama3 with parameters")
    # parser.add_argument('--output_dir', type=str, default="outputs_Llama-3-ELYZA-JP-8B", help="出力ディレクトリ")
    # parser.add_argument('--data_files', type=str, default="./train_dataset.json", help="学習データ(json)のパス")
    # args = parser.parse_args()

def model_train(model_name, dataset_path, output_dir):

    max_seq_length = 2048
    dtype = None
    load_in_4bit = False
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

    EOS_TOKEN = tokenizer.eos_token
    def formatting_prompts_func(examples):
        instructions = examples["instruction"]
        inputs       = examples["input"]
        outputs      = examples["output"]
        texts = []
        for instruction, input, output in zip(instructions, inputs, outputs):
            text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
            texts.append(text)
        return { "text" : texts }
    pass

    from datasets import load_dataset
    dataset = load_dataset("json", data_files = dataset_path, split = "train")
    dataset = dataset.map(formatting_prompts_func, batched = True)

    from trl import SFTTrainer
    from transformers import TrainingArguments

    model = FastLanguageModel.get_peft_model(
        model,
        r = 16, # 8, 16, 32, 64, 128
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = dataset,
        dataset_text_field = "text",
        max_seq_length = max_seq_length,
        dataset_num_proc = 2,
        packing = False,
        args = TrainingArguments(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            max_steps = 60,
            learning_rate = 2e-4,
            fp16 = not torch.cuda.is_bf16_supported(),
            bf16 = torch.cuda.is_bf16_supported(),
            logging_steps = 1,
            optim = "adamw_8bit",
            weight_decay = 0.01,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = output_dir,
        ),
    )


    trainer_stats = trainer.train()
    # モデルの保存
    trainer.save_model(f"{output_dir}/finetuned_model")
    # モデルの保存後、トークナイザーも保存
    tokenizer.save_pretrained(f"{output_dir}/finetuned_model")
