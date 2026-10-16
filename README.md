# Gift Card review sentiment classifier

An LLM-based binary classifier (not a locally trained model). Reads `Gift_Cards.jsonl` and sends only `title`, `text`, and `rating` to the OpenAI Responses endpoint using structured outputs. Defaults to `gpt-4.1-mini`; use `--model` to change it.

## Setup

Dependencies are listed in requirements.txt. For an isolated installation with uv:

```bash
uv venv .venv
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
```

Set `OPENAI_API_KEY` in your environment using your normal secure secret-management process; do not save keys in this repository or paste them into chat. The examples below use `python`; use `.venv/Scripts/python.exe` if you installed in the virtual environment.

## Run

```bash
# Validate every source record; no API calls and no output file.
python sentiment_classifier.py --dry-run --limit 0

# Classify the first 10 records using OpenAI (billable).
python sentiment_classifier.py --limit 10 --output sentiments_10.jsonl

# Default: first 100 records.
python sentiment_classifier.py

# Explicit opt-in to classify every record (one billable request per review).
python sentiment_classifier.py --limit 0 --output sentiments_all.jsonl

# Tests use a mocked API boundary; no credentials or charges.
python -m unittest test_sentiment_classifier -v
```

## Classification policy

The LLM reads title and text, using rating as supporting evidence. Clear written sentiment takes priority over a conflicting rating. Mixed sentiment uses the dominant opinion. Neutral/tied or empty text uses rating as a forced binary fallback: 4–5 positive, 1–3 negative. Three-star reviews are not intrinsically negative: this is an explicit tie-break because only two labels were requested.

Output is JSON Lines with `source_line`, `title`, `text`, `rating`, `sentiment`, `reason`, and `model`. The label schema permits only `positive` or `negative`. No results are fabricated on failure. Partial results are flushed after each review; execution stops on an error. Existing output files are never overwritten. There is no automatic resume: choose a new output path on rerun (completed rows would be billed again).

The full source is streamed instead of loaded into memory. API timeout is 60 seconds with up to three SDK retries for retryable errors. Review contents are treated as data rather than instructions. Only selected fields leave the machine; `store=False` is requested, which is not a guarantee of zero provider retention.

## Limits and evaluation

No live predictions have been verified without API credentials. Unit tests check integration behavior with a mocked OpenAI client, not actual model accuracy. Validate a human-labeled sample before using predictions for decisions. Comparing against rating-derived labels is not independent accuracy evaluation because the model also receives rating. LLM results may vary across runs.

References: OpenAI structured outputs guide (`developers.openai.com/api/docs/guides/structured-outputs`) and GPT-4.1 mini model documentation (`developers.openai.com/api/docs/models/gpt-4.1-mini`).
