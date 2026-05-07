import os
import json
from pathlib import Path

from torch import mode


# 以下の形式のjsonを読み込む
# {
# 	"message": "わあ、みんなと暮らすこの家には冷蔵庫のスペースも重要なアート作品みたいなものだよね！でも、それぞれ好きなものや必要なものが違うから、みんなの好みや事情を尊重しながら、どうやって最高にうまくルールを作れると思う？君のクリエイティブなアイデアを聞かせてほしいな！",
# 	"response": "冷蔵庫のスペースの話は先ほども言いましたが、食品の種類によってスペースを分けるのか、一人一人に一定のスペースを与えるか",
# 	"prediction": "What a fascinating question! I think, each personの個性や好みを尊重するため、冷蔵庫のルールを「Art Gallery」に例えるのはとても面白いアイデアです。\n\nまず、冷蔵庫を「Art Gallery」に例えるなら、各自の好きなものや必要なものを「作品」と捉えます。各「作品」には、作者の名前、作品の説明、作者の好みやこだわりを記した「アーティストステートメント」を付けることで、共有するメンバーが理解し、尊重することができます。\n\nルール作りのプロセスは、以下の通りです:\n\n1. 各自の「作品」について、作者がアーティストステートメントを書きます。\n2. メンバー全員が、各「作品」を冷蔵庫内に配置します。\n3. 冷蔵庫の「展示」が完了した後、メンバーが、他の「作品」について理解し、尊重するために、各「作品」について、作者にインタビューします。\n4. インタビューを通じて、共有するメンバーが、各「作品」についての理解を深め、共有する価値観やルールを明確にします。\n5. 冷蔵庫の「Art Gallery」に、共有するメンバーが、ルールや注意事項を記した「Art Catalog」や「Artist's Statement」等の「作品」を追加し、共有します。\n\nこのプロセスを通じて、冷蔵庫の「Art Gallery」は、メンバーの個性や好みを尊重し、共有する価値観やルールを明確にし、メンバーが、共に快適に暮らせる空間を創造することができます。",
# 	"scores": {
# 		"wmd": 0.7617410704756408,
# 		"sentence_similarity": 0.04741690307855606,
# 		"normalized_length_similarity": 0.09090909090909091,
# 		"bleu": 0.0020921397938141687,
# 		"ttr": 0.29362880886426596,
# 		"history_similarity": 0.19106563925743103
# 	}
# },

def summarize_score(base_dir, model_name, lang):
	model_lang_dir = Path(base_dir) / model_name.replace("/", "_") / lang
	# 言語ごとにスコアの平均値を計算
	lang_scores = {
	"wmd": 0.0,
	"sentence_similarity": 0.0,
	"normalized_length_similarity": 0.0,
	"bleu": 0.0,
	"ttr": 0.0,
	"history_similarity": 0.0
	}
	num_person = 0
	
	for json_file in model_lang_dir.glob("*.json"):
		with open(json_file, "r") as f:
			data = json.load(f)
		num_person += 1
		
		# まず各jsonファイルごとに平均をとる
		person_scores = {
			"wmd": 0.0,
			"sentence_similarity": 0.0,
			"normalized_length_similarity": 0.0,
			"bleu": 0.0,
			"ttr": 0.0,
			"history_similarity": 0.0
		}
		for chat in data["scores"]:
			# chat["scores"] に各スコアが入っている
			person_scores["wmd"] += chat["scores"]["wmd"] if chat["scores"]["wmd"] != float("inf") else 1
			person_scores["sentence_similarity"] += chat["scores"]["sentence_similarity"]
			person_scores["normalized_length_similarity"] += chat["scores"]["normalized_length_similarity"]
			person_scores["bleu"] += chat["scores"]["bleu"]
			person_scores["ttr"] += chat["scores"]["ttr"]
			person_scores["history_similarity"] += chat["scores"]["history_similarity"]
		num_chats = len(data["scores"])
		for key in person_scores:
			person_scores[key] /= num_chats
		
		# langごとに、json全体の平均をとる
		lang_scores["wmd"] += person_scores["wmd"]
		lang_scores["sentence_similarity"] += person_scores["sentence_similarity"]
		lang_scores["normalized_length_similarity"] += person_scores["normalized_length_similarity"]
		lang_scores["bleu"] += person_scores["bleu"]
		lang_scores["ttr"] += person_scores["ttr"]
		lang_scores["history_similarity"] += person_scores["history_similarity"]
	for key in lang_scores:
		lang_scores[key] /= num_person
	# 結果を表示
	# wmd, sentence_similarity, bleu, normalized_length_similarity, ttr, history_similarityの順番
	# latexフォーマットで、4桁で表示
	# 例：& 0.247 & 0.402 & 0.0117 & 0.346 & 0.800 & 0.398 
	# 言語の人数も表示
	print(f"{lang, model_name, num_person} & {lang_scores['wmd']:.3f} & {lang_scores['sentence_similarity']:.3f} & {lang_scores['bleu']:.4f} & {lang_scores['normalized_length_similarity']:.3f} & {lang_scores['ttr']:.3f} & {lang_scores['history_similarity']:.3f} \\\\")


models = [
	"elyza/Llama-3-ELYZA-JP-8B",
	"meta-llama/Llama-3.1-8B-Instruct",
	"Qwen/Qwen3-14B"
]
for model_name in models:
	for lang in ["jp", "en", "cn"]:
		summarize_score(
			base_dir="directory_for_model_response",
			model_name=model_name,
			lang=lang
		)