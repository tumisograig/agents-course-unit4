# Unit 4 final assignment — HF Agents Course

Local free-submission setup: no Hugging Face PRO and no paid Space compute.
The scoring server (`https://agents-course-unit4-scoring.hf.space`) is a plain
REST API: `GET /questions`, `GET /files/{task_id}`, `POST /submit`
(`{username, agent_code, answers}`; `agent_code` is this repo's public URL).

## Setup

```bash
pip install -r requirements.txt
# or, the course's own line:
pip install "smolagents[litellm,toolkit]" requests pandas openpyxl pypdf
```

Model: a free OpenRouter route by default (`inclusionai/ling-3.2-flash:free`).
Override with `--model` or the `AGENT_MODEL` env var. The key is read from
`~/.config/graig/openrouter.key` (or `OPENROUTER_API_KEY`); it is never printed
or committed.

## Run

```bash
python submit_agent.py            # dry run: answers land in answers.json
python submit_agent.py --submit   # POST to /submit and print the score
```

Re-run single questions with `--only <task_id-prefix> --force`.

## Files

- `agent.py` — the agent (smolagents CodeAgent + web search, webpage reader,
  YouTube transcripts, local whisper transcription, python executor). This is
  the code the public repo link points at.
- `submit_agent.py` — the local runner/submitter (dry run by default).
- `answers.json`, `files/` — generated caches, not committed.

Certificate: 30%+ (6 of 20) passes Unit 4.
