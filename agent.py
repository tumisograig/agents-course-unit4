"""Unit 4 final assignment agent (HF Agents Course).

A smolagents CodeAgent that answers GAIA-style questions with web search,
webpage reading, YouTube transcripts, local audio transcription and a
sandboxed Python executor (for the .py and .xlsx attachments).

The public repo link for this file is what goes into the scoring server's
`agent_code` field. No credentials live in this repo: the OpenRouter key is
read from ~/.config/graig/openrouter.key (or OPENROUTER_API_KEY) at runtime.
"""

import os
from pathlib import Path

from smolagents import CodeAgent, FinalAnswerTool, LiteLLMModel, VisitWebpageTool

try:  # newer smolagents
    from smolagents import WebSearchTool as SearchTool
except ImportError:  # older smolagents
    from smolagents import DuckDuckGoSearchTool as SearchTool

from smolagents import Tool

API_URL = "https://agents-course-unit4-scoring.hf.space"
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "openrouter/particle-ai/particle-1:free")
KEY_FILE = Path.home() / ".config" / "graig" / "openrouter.key"

AUTHORIZED_IMPORTS = [
    "pandas",
    "openpyxl",
    "pypdf",
    "json",
    "re",
    "math",
    "statistics",
    "datetime",
    "collections",
    "itertools",
]

ANSWER_RULES = """\
Answer rules (follow strictly):
- Answer the question directly. No preamble, no explanation, no citations.
- If the answer is a list, give a comma-separated list, alphabetized unless the
  question says otherwise.
- Numbers: no thousands separators, no units unless the question asks for them.
- If asked for a name, give only the name part the question asks for.
- Use the tools to verify facts; never invent numbers, names or titles.
- If you cannot determine the answer after exhausting your tools, give your
  single best guess in the exact requested format. Never refuse or write
  "I don't know".
"""


class YouTubeTranscriptTool(Tool):
    """Text transcript of a YouTube video via youtube-transcript-api."""

    name = "youtube_transcript"
    description = ("Returns the English text transcript of a YouTube video. "
                   "Input: the video URL or video id.")
    inputs = {"url": {"type": "string", "description": "YouTube URL or video id"}}
    output_type = "string"

    def forward(self, url: str) -> str:
        from youtube_transcript_api import YouTubeTranscriptApi

        video_id = url.split("v=")[-1].split("&")[0].split("/")[-1]
        segments = YouTubeTranscriptApi().fetch(video_id, languages=["en"])
        text = " ".join(s.text for s in segments)
        return text[:6000]


class TranscribeAudioTool(Tool):
    """Local transcription of an audio file with faster-whisper (tiny, int8)."""

    name = "transcribe_audio"
    description = ("Transcribes a local audio file (.mp3/.wav/.m4a) to English text. "
                   "Input: the path to the audio file.")
    inputs = {"path": {"type": "string", "description": "Path to the audio file"}}
    output_type = "string"

    def forward(self, path: str) -> str:
        from faster_whisper import WhisperModel

        model = WhisperModel("tiny.en", device="cpu", compute_type="int8")
        segments, _info = model.transcribe(path)
        return " ".join(s.text for s in segments).strip()


def load_openrouter_key() -> str:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key and KEY_FILE.exists():
        key = KEY_FILE.read_text().strip()
    if not key:
        raise RuntimeError(
            "No OpenRouter key: set OPENROUTER_API_KEY or create "
            f"{KEY_FILE} (chmod 600)."
        )
    return key


class Unit4Agent:
    """Callable agent: agent(question) or agent(question, file_path) -> str."""

    def __init__(self, model_id: str | None = None):
        self.model_id = model_id or DEFAULT_MODEL
        model = LiteLLMModel(
            model_id=self.model_id,
            api_base="https://openrouter.ai/api/v1",
            api_key=load_openrouter_key(),
            reasoning_effort="none",  # particle-1:free answers in content, not reasoning
        )
        self.agent = CodeAgent(
            tools=[
                SearchTool(),
                VisitWebpageTool(),
                YouTubeTranscriptTool(),
                TranscribeAudioTool(),
                FinalAnswerTool(),
            ],
            model=model,
            max_steps=12,
            additional_authorized_imports=AUTHORIZED_IMPORTS,
            verbosity_level=0,
        )

    def __call__(self, question: str, file_path: str | None = None) -> str:
        task = question.strip()
        if file_path:
            p = Path(file_path)
            task += (
                f"\n\n[Attachment] A file is saved for you at: {p}\n"
                "Read it with the python executor (open it, and for images note "
                "you cannot see pixels, so reason from the question text) or "
                "transcribe it with transcribe_audio if it is audio.\n"
            )
        task += "\n\n" + ANSWER_RULES
        answer = self.agent.run(task)
        return str(answer).strip()


# Name kept compatible with the course template so any grader-side check
# that imports BasicAgent keeps working.
BasicAgent = Unit4Agent


if __name__ == "__main__":
    agent = Unit4Agent()
    print(agent("If you understand this sentence, write the opposite of the word 'left', as the answer."))
