#!/usr/bin/env python3
"""Local submission runner for the HF Agents Course Unit 4 final assignment.

Fetches the questions from the scoring server, downloads each question's
attachment, runs the agent in agent.py, caches answers in answers.json and
(optionally, with --submit) POSTs everything to /submit.

Default is a dry run: nothing is sent to the server.

Examples:
  python submit_agent.py                          # dry run, all questions
  python submit_agent.py --only 8e867cd7 --refresh # re-run one question
  python submit_agent.py --submit                 # cache + POST to /submit
"""

import argparse
import json
import pathlib
import sys
import time

import requests

from agent import API_URL, Unit4Agent

HERE = pathlib.Path(__file__).resolve().parent
ANSWERS_FILE = HERE / "answers.json"
FILES_DIR = HERE / "files"
CODE_URL = "https://github.com/tumisograig/agents-course-unit4/tree/main"


def fetch_questions() -> list[dict]:
    r = requests.get(f"{API_URL}/questions", timeout=30)
    r.raise_for_status()
    return r.json()


def download_file(task_id: str, file_name: str) -> pathlib.Path:
    dest_dir = FILES_DIR / task_id
    dest = dest_dir / file_name
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(f"{API_URL}/files/{task_id}", timeout=60)
    r.raise_for_status()
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def load_answers() -> dict:
    if ANSWERS_FILE.exists():
        return json.loads(ANSWERS_FILE.read_text())
    return {}


def save_answers(answers: dict) -> None:
    ANSWERS_FILE.write_text(json.dumps(answers, indent=2, sort_keys=True))


def run(args) -> tuple[dict, list[dict]]:
    questions = fetch_questions()
    print(f"Fetched {len(questions)} questions from {API_URL}/questions", flush=True)

    answers = {} if args.refresh else load_answers()
    todo = [q for q in questions if q["task_id"] not in answers]
    if args.only:
        todo = [q for q in questions if q["task_id"].startswith(args.only)]
    if args.force:
        answers = {k: v for k, v in answers.items()
                   if not any(q["task_id"].startswith(args.only or "") for q in todo)}
        todo = [q for q in questions
                if (not args.only or q["task_id"].startswith(args.only))]

    agent = Unit4Agent(model_id=args.model)
    log = []
    for i, q in enumerate(todo, 1):
        task_id, text, file_name = q["task_id"], q["question"], q.get("file_name", "")
        print(f"[{i}/{len(todo)}] {task_id} {text[:70]!r}", flush=True)
        started = time.time()
        try:
            file_path = str(download_file(task_id, file_name)) if file_name else None
            answer = agent(text, file_path)
        except Exception as exc:  # keep the run going; record the failure
            answer = f"AGENT ERROR: {exc}"
        answers[task_id] = answer
        save_answers(answers)
        log.append({"task_id": task_id, "question": text,
                    "answer": answer, "seconds": round(time.time() - started, 1)})
        print(f"      -> {answer[:120]} ({log[-1]['seconds']}s)", flush=True)

    print(f"\n{len(answers)}/{len(questions)} answers ready in {ANSWERS_FILE}")
    return answers, log


def submit(args, answers: dict) -> None:
    payload = {
        "username": args.username,
        "agent_code": args.code_url,
        "answers": [{"task_id": t, "submitted_answer": a} for t, a in sorted(answers.items())],
    }
    print(f"Submitting {len(payload['answers'])} answers as {args.username} ...", flush=True)
    r = requests.post(f"{API_URL}/submit", json=payload, timeout=120)
    r.raise_for_status()
    out = r.json()
    print(f"\nScore: {out.get('score')}%  ({out.get('correct_count')}/{out.get('total_attempted')})")
    print(f"Message: {out.get('message')}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--username", default="Tumisograig", help="Hugging Face username")
    p.add_argument("--code-url", default=CODE_URL, help="Public URL of the agent code")
    p.add_argument("--model", default=None, help="OpenRouter model id (default: agent.py's)")
    p.add_argument("--submit", action="store_true", help="POST to /submit (default: dry run)")
    p.add_argument("--only", default=None, help="run only task ids starting with this")
    p.add_argument("--refresh", action="store_true", help="ignore the cached answers")
    p.add_argument("--force", action="store_true", help="re-run the --only questions even if cached")
    args = p.parse_args()

    answers, _log = run(args)
    if args.submit:
        submit(args, answers)
    else:
        print("Dry run only: pass --submit to POST to the scoring server.")


if __name__ == "__main__":
    sys.exit(main())
