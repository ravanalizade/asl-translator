"""
backend/gemini_client.py
------------------------
Wraps Gemini 2.5 Flash for ASL gloss → natural English conversion.

Triggered every N accumulated words or after a timeout.
"""

import os
import asyncio
import google.generativeai as genai


SYSTEM_PROMPT = """You are an ASL-to-English translator. You receive a sequence of ASL sign \
glosses (individual words detected from American Sign Language) and must convert \
them into a natural, grammatically correct English sentence. ASL grammar differs \
from English — ASL uses topic-comment structure, often drops articles and \
auxiliary verbs, and may reorder words. Interpret the intent and produce fluent \
English. Reply with ONLY the English sentence, nothing else."""

FEW_SHOT_EXAMPLES = [
    ("HELLO HOW YOU", "Hello, how are you?"),
    ("THANK-YOU HELP ME PLEASE", "Thank you, please help me."),
    ("MY NAME WHAT", "What is my name?"),
    ("YOU WANT EAT", "Do you want to eat?"),
    ("I SORRY LATE", "I'm sorry I'm late."),
]


class GeminiClient:
    """
    Async Gemini 2.5 Flash client for gloss → English translation.

    Usage:
        client = GeminiClient()
        sentence = await client.translate(["HELLO", "HOW", "YOU"])
    """

    def __init__(self, api_key: str = None):
        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            raise ValueError(
                "GEMINI_API_KEY not set. Add it to your .env file.\n"
                "Get a free key at: https://aistudio.google.com/"
            )
        genai.configure(api_key=key)
        self._model = genai.GenerativeModel(
            model_name="gemini-2.5-flash",
            system_instruction=SYSTEM_PROMPT,
        )
        self._build_chat_history()

    def _build_chat_history(self):
        """Pre-load few-shot examples into the chat context."""
        self._history = []
        for gloss, english in FEW_SHOT_EXAMPLES:
            self._history.append({"role": "user",  "parts": [gloss]})
            self._history.append({"role": "model", "parts": [english]})

    async def translate(self, words: list[str]) -> str:
        """
        Convert a list of ASL glosses to a natural English sentence.

        Args:
            words: list of uppercase ASL glosses, e.g. ["HELLO", "HOW", "YOU"]

        Returns:
            English sentence string, e.g. "Hello, how are you?"
        """
        if not words:
            return ""

        gloss_input = " ".join(w.upper() for w in words)

        try:
            # Run in executor so it doesn't block the async event loop
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                lambda: self._model.generate_content(
                    contents=self._history + [{"role": "user", "parts": [gloss_input]}],
                    generation_config=genai.GenerationConfig(
                        temperature=0.3,   # low temp for consistent grammar correction
                        max_output_tokens=128,
                    ),
                )
            )
            return response.text.strip()

        except Exception as e:
            print(f"[Gemini] Error: {e}")
            # Graceful fallback: just join the words
            return " ".join(w.capitalize() for w in words) + "."

    def translate_sync(self, words: list[str]) -> str:
        """Synchronous version for non-async contexts."""
        if not words:
            return ""
        gloss_input = " ".join(w.upper() for w in words)
        try:
            response = self._model.generate_content(
                contents=self._history + [{"role": "user", "parts": [gloss_input]}],
                generation_config=genai.GenerationConfig(temperature=0.3, max_output_tokens=128),
            )
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini] Error: {e}")
            return " ".join(w.capitalize() for w in words) + "."
