# predictors/cot_predictor.py
class CoTPredictor:
    def __init__(self, llm):
        self.llm = llm

    # -----------------------------------------
    # 内部推論①：文体・態度の抽出（非公開）
    # -----------------------------------------
    def infer_style(self, train_data: str) -> dict:
        prompt = f"""
        You are analyzing past replies written by speaker A.
        Infer A's typical communication style using the criteria below.

        CRITICAL RULES:
        - DO NOT restate, summarize, or repeat the definitions.
        - DO NOT output explanations, examples, or placeholder text.
        - DO NOT output anything except the final labels.
        - If you cannot decide, choose the closest option.

        IMPORTANT:
        - Base your judgment ONLY on text written by speaker A.
        - Ignore any questions, prompts, or messages written by other speakers.
        - Use the definitions strictly as written.
        - Output ONLY the labels using '=' exactly.
        - Do NOT explain your reasoning or include any extra text.

        --------------------------------
        DEFINITIONS

        tone (wording style only):
        - formal  = polite wording, indirect expressions, honorifics, or restrained phrasing
        - casual  = informal wording, direct expressions, or conversational tone

        length (response length):
        - short   = 1 sentence AND fewer than 15 words
        - medium  = 2–3 sentences OR 15–40 words
        - long    = 4 or more sentences OR more than 40 words
        If criteria conflict, prioritize sentence count over word count.

        emotion (emotional intensity only):
        - low     = emotionally neutral or factual wording, no emphasis or affective expressions
        - medium  = mild emotions (e.g., friendliness or soft concern), limited emphasis
        - high    = strong emotions, expressive wording, emotional intensifiers, or exclamations
        Ignore politeness or interpersonal closeness when judging emotion.

        distance (interpersonal stance only):
        - close   = friendly, personal, or empathetic engagement
        - neutral = polite but impersonal, task-focused, socially standard
        - distant = detached, minimal engagement, or formal without warmth
        Ignore emotional intensity when judging distance.

        NOTE:
        - tone and distance are independent dimensions.
        - A response can be formal and close, or formal and distant.

        --------------------------------
        Responses by speaker A:
        {train_data}

        --------------------------------
        Output format (exactly):
        tone=<formal/casual>
        length=<short/medium/long>
        emotion=<low/medium/high>
        distance=<close/neutral/distant>
        """

        raw = self.llm.generate_simple(prompt)
        try:
            style = {}
            for line in raw.splitlines():
                if "=" in line:
                    k, v = line.split("=")
                    style[k.strip()] = v.strip()
        except:
            style = raw

        return style

    # -----------------------------------------
    # 最終予測（可視出力）
    # -----------------------------------------
    def predict(self, question: str, train_data: str | None, style: dict | str | None):

        system_prompt = f"""
        You have joined the share house as a new resident.
        DaVinci, Donatello, Michelangelo, and Raffaello are members of the share house.

        IMPORTANT STYLE RULES:
        - When learning the writing style for your reply ("A"),
        **use ONLY the examples labeled "A:"** from previous interactions.
        - **Do NOT imitate or be influenced by any "Q" lines**
        (their tone, length, emotion, or punctuation must be ignored).
        - Learn style exclusively from "A:" responses.

        BEFORE writing your reply:
        - Think internally about tone, politeness, emotion, and length
        that best match A's past responses.
        - Do NOT explain your reasoning.

        FINAL OUTPUT RULES:
        - Output ONLY the final predicted response.
        - Do NOT include labels like "A:".
        - Do NOT include explanations, reasoning, or quotes.
        """

        # ★ CoT由来の制約を追記（reasoningは出力させない）
        if style:
            if isinstance(style, dict):
                system_prompt += f"""
                Constraints inferred from past responses:
                - tone: {style.get('tone')}
                - length: {style.get('length')}
                - emotion: {style.get('emotion')}
                - distance: {style.get('distance')}

                Think internally but do not explain your reasoning.
                """
            elif isinstance(style, str):
                system_prompt += f"""
                {style}

                Think internally but do not explain your reasoning.
                """
            else:
                system_prompt = None


        user_prompt = (
            f"previous interactions:\n{train_data}\n\n"
            if train_data else ""
        )
        user_prompt += f"Please predict the user's response to the following message:\n{question}"

        print(f"system_prompt: \n{system_prompt}")
        print(f"user_prompt: \n{user_prompt}")

        return self.llm.generate(system_prompt, user_prompt)
