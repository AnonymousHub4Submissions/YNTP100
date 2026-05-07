# anonymous_100

A collection of 100 anonymized multi-turn conversational data samples used for the YNTP-100 multilingual benchmark.

## Structure

```
anonymous_100/
├── cn/          # Chinese conversation samples (cn_1.json – cn_N.json)
├── en/          # English conversation samples (en_1.json – en_N.json)
└── jp/          # Japanese conversation samples (jp_1.json – jp_N.json)
```

## Data Format

Each `.json` file contains a multi-day dialogue between a user and multiple NPC agents (DaVinci, Donatello, Michelangelo, Raffaello). Conversations are organized by day, with each turn recording the speaker, the agent's message, and the user's response.

```json
{
  "day_1": [
    {
      "speaker": "DaVinci",
      "message": "<agent utterance>",
      "response": "<user response>"
    },
    ...
  ],
  ...
}
```


## Usage

These samples are intended for evaluation purposes only. Each file represents one participant's full conversation session across multiple days in the ShareHouse scenario.
