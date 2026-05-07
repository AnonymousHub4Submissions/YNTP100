import os
import json
from pathlib import Path
import datetime
import numpy as np
from numpy import dot
from numpy.linalg import norm
from functools import lru_cache

from sudachipy import dictionary, tokenizer as sudachi_tokenizer
import nltk
# nltk.download('punkt_tab')

from nltk import bleu_score
from nltk.translate.bleu_score import SmoothingFunction
import jieba
from nltk.tokenize import word_tokenize
import gensim
from sentence_transformers import SentenceTransformer, util

#* https://qiita.com/Hironsan/items/513b9f93752ecee9e670のモデルを使用しています
# mymodel-train\day5_prediction\vecs\fastText_model.vecを使用
wmd_model = gensim.models.KeyedVectors.load_word2vec_format('mymodel_train/mymodel-train/day5_prediction/vecs/fastText_model.vec', binary=False)
sent_sim_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
history_model = SentenceTransformer("all-mpnet-base-v2")
from sentence_transformers import SentenceTransformer, util




# smoothing function を用いると短文での BLEU が 0 になるのを緩和できる
smoothing_function = SmoothingFunction().method1

# Reuse tokenizer instances to avoid repeated construction cost
_sudachi_tokenizer = dictionary.Dictionary().create()
_sudachi_mode = sudachi_tokenizer.Tokenizer.SplitMode.C


@lru_cache(maxsize=100_000)
def tokenize_japanese(text):
    tokens = [m.surface() for m in _sudachi_tokenizer.tokenize(text, _sudachi_mode)]
    return tokens


@lru_cache(maxsize=100_000)
def tokenize_chinese(text):
    tokens = list(jieba.cut(text))
    # print("tokenizer (zh): jieba")
    return tokens

@lru_cache(maxsize=100_000)
def tokenize_english(text):
    
    tokens = word_tokenize(text)
    # print("tokenizer (en): nltk word_tokenize")
    return tokens


#* [0, 1], higher is better
@lru_cache(maxsize=50_000)
def _cached_sent_vector(text: str):
    # Unit-normalized numpy vector (float32)
    return sent_sim_model.encode(text, normalize_embeddings=True).astype(np.float32)


def sentence_similarity(correct_answer, predicted_answer):
    v1 = _cached_sent_vector(correct_answer)
    v2 = _cached_sent_vector(predicted_answer)
    # With normalized vectors, cosine is just dot product
    return float(np.dot(v1, v2))

#* [0, ], lower is better
@lru_cache(maxsize=20_000)
def _cached_wmdistance(a: str, b: str) -> float:
    return float(wmd_model.wmdistance(a, b))


def wmdistance(correct_answer, predicted_answer):
    return _cached_wmdistance(correct_answer, predicted_answer)

#* [0, 1], higher is better
def normalized_length_similarity(correct_answer, predicted_answer):
    len1 = len(correct_answer.strip())
    len2 = len(predicted_answer.strip())
    return float(min(len1, len2) / max(len1, len2))

#* ユーザーの応答履歴から作ったベクトルと生成した応答のベクトルの類似度
@lru_cache(maxsize=50_000)
def _cached_hist_vector(text: str):
    return history_model.encode(text, normalize_embeddings=True).astype(np.float32)


def history_vector_similarity(user_history, text):
    # Average user's history vectors (individually normalized)
    if user_history:
        hist_vecs = [_cached_hist_vector(t) for t in user_history]
        history_vector = np.mean(hist_vecs, axis=0)
    else:
        history_vector = _cached_hist_vector("") if hasattr(history_model, "encode") else np.zeros_like(_cached_hist_vector(text))

    gen_vec = _cached_hist_vector(text)

    def cosine_similarity(a, b):
        return dot(a, b) / (norm(a) * norm(b))

    style_sim = cosine_similarity(history_vector, gen_vec)
    return float(style_sim)


def bleu(text_ans, text_pred, lang):
    if lang == 'ja' or lang == 'jp':
        refs_tokens = tokenize_japanese(text_ans)
        hyp_tokens = tokenize_japanese(text_pred)
    elif lang == 'zh' or lang == 'cn':
        refs_tokens = tokenize_chinese(text_ans)
        hyp_tokens = tokenize_chinese(text_pred)
    elif lang == 'en':
        refs_tokens = tokenize_english(text_ans)
        hyp_tokens = tokenize_english(text_pred)
    else:
        raise ValueError("Unsupported language for BLEU calculation.")

    bleu_token_level = bleu_score.sentence_bleu([refs_tokens], hyp_tokens, smoothing_function=smoothing_function)
    return float(bleu_token_level)

def ttr(text, lang):
    if lang == 'ja' or lang == 'jp':
        tokens = tokenize_japanese(text)
    elif lang == 'zh' or lang == 'cn':
        tokens = tokenize_chinese(text)
    elif lang == 'en':
        tokens = tokenize_english(text)
    else:
        raise ValueError("Unsupported language for TTR calculation.")
    
    if not tokens:
        return 0
    types = set(tokens)
    return float(len(types) / len(tokens))


def score_json(method, using_model, lang, model_json_path):
    print("Processing file:", model_json_path)
    filename = Path(model_json_path).stem.replace('.', '_')


    # ディレクトリがなければ作成
    output_dir = Path("score") / using_model.replace("/", "_") / lang
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{filename}.json"


    results = []
    #* チャットのデータ
    with open(model_json_path, "r") as f:
        all_day = json.load(f)
        
            
        output_json = {
            "metadata": {
                "model": using_model,
                "method": method,
                "date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                #! 以下の項目はmethodごとで多少変わるかもしれない
                #! DPOの場合の例
                "instruction": all_day["metadata"][0]["instruction"],
                "chat_template": all_day["metadata"][0]["chat_template"]
            },
            "scores": []
            }

        #* ユーザーの応答のみのリストも用意
        user_history = []
        for i in range(1, 5):
            day_data = all_day[f"day_{i}"]
            for interaction in day_data:
                user_history.append(interaction["response"])
        # print("User history:", user_history)
        
        #* day4までをプロンプトに入れて、dayの応答を予測
        #* まず質問と正解の応答のペアを用意
        day5 = all_day[f"day_5"]

    for interaction in day5:
        question = interaction["message"]
        correct_answer = interaction["response"].strip()
        predicted_answer = interaction["prediction"].strip()
        

        wmd1 = wmdistance(correct_answer, predicted_answer)
        sim1 = sentence_similarity(correct_answer, predicted_answer)
        len_sim1 = normalized_length_similarity(correct_answer, predicted_answer)

        bleu_value = bleu(correct_answer, predicted_answer, lang)
        ttr_value = ttr(predicted_answer, lang)
        # history_sim_ans = history_vector_similarity(user_history, correct_answer)
        history_sim_pred = history_vector_similarity(user_history, predicted_answer)
        # history_sim_pred = f"{history_sim_pred} / {history_sim_ans}"
        # 結果をoutput_json["scores"]に追加
        output_json["scores"].append({
            "message": question,
            "response": correct_answer,
            "prediction": predicted_answer,
            "scores": {
                "wmd": float(wmd1),
                "sentence_similarity": float(sim1),
                "normalized_length_similarity": float(len_sim1),
                "bleu": float(bleu_value),
                "ttr": float(ttr_value),
                "history_similarity": float(history_sim_pred)
            }
        })
        
    #* filenameはjson_fileのファイル名から拡張子を除いたもの


    # JSONファイルとして保存
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_json, f, ensure_ascii=False, indent=4)
    print("Saved score to:", output_path)

