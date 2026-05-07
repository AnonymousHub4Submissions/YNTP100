
import os
import json
from pathlib import Path
from openai import OpenAI
client = OpenAI()
openai_api_key = os.getenv("OPENAI_API_KEY")


import datetime
import copy



def predict_response_with_data(using_model, train_data, question, get_prompt=False):
    system_prompt = f"""You have joined the share house as a new resident. DaVinci, Donatello, Michelangelo, and Raffaello are members of the share house. I will provide you with the previous exchanges from the conversation. Here, "A" refers to your reply. Please carefully observe the tone, attitude, values, and other cues. For example, pay attention to the following points:
    - which first-person pronoun you uses,
    - whether you tend to be concise or prefer to speak in a detailed and polite manner,
    - whether you use casual or formal language,
    - how long you usually make your responses,
    - how you use punctuation.
    Especially, please pay attention to the length of your responses.
    Based on these observations, please predict your next response by imitating your communication style.
    """

    user_prompt = f"""previous interactions :
    {train_data}
    Please predict your response to the following message:{question}
    """
    print("user_prompt:", user_prompt)
    
    if get_prompt:
        return system_prompt, user_prompt

    response = client.chat.completions.create(
    model=using_model,
    messages=[
        {
            "role": "developer",
            "content": system_prompt
        },
        {
            "role": "user",
            "content": user_prompt
        }
    ]
    )

    return response.choices[0].message.content



def predict_json(input_path, using_model, lang):
    print("Processing file:", input_path)

    filename = Path(input_path).stem.replace('.', '_')

    
    # ディレクトリがなければ作成
    output_dir = Path("response") / using_model.replace("/", "_") / lang
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename}.json"


    #* チャットのデータ
    with open(input_path, "r") as f:
        all_day = json.load(f)
        
        #! 入力ファイルに追記して保存する
        output_json = copy.deepcopy(all_day)

        train_day_list = []
        for i in range(1, 5):
            day_data = all_day[f"day_{i}"]
            for interaction in day_data:
                train_day_list.append(f"Q({interaction['speaker']}): {interaction['message']}\nA: {interaction['response']}")
        print("Train day list:", train_day_list)
        
        #* 1つのQ-Aごとに改行したい
        train_day = "\n\n".join(train_day_list)
        print("Train day:", train_day)

        #* ユーザーの応答のみのリストも用意
        user_history = []
        for i in range(1, 5):
            day_data = all_day[f"day_{i}"]
            for interaction in day_data:
                user_history.append(interaction["response"])
        
        #* day4までをプロンプトに入れて、dayの応答を予測
        #* まず質問と正解の応答のペアを用意
        try:
            day5 = all_day["day_5"]
        except Exception:
            print(f"{filename} is missing day5 data")
            return
        
        day5_data = {}
        for interaction in day5:
            question = interaction["message"]
            correct_answer = interaction["response"]
            day5_data[question] = correct_answer

        print(day5_data)


    for interaction in day5:
        question = interaction["message"]
        correct_answer = interaction["response"].strip()

        predicted_answer = predict_response_with_data(using_model, train_day, question)
        predicted_answer = predicted_answer.strip()
        #* 結果の書き込み
        output_json["day_5"][output_json["day_5"].index(interaction)]["prediction"] = predicted_answer
       
    
    #* promptも保存
    output_json["metadata"] = {
        "model": using_model,
        "method": "Prompt Engineering-Few-Shot",
        "predicted_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "system_prompt": predict_response_with_data(using_model, "train_day", "question", get_prompt=True)[0],
        "user_prompt": predict_response_with_data(using_model, "train_day", "question", get_prompt=True)[1],
        },


    # JSONファイルとして保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
    print("Saved to:", output_path)

