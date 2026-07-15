
"""
groq_client.py

Reusable Groq LLM client for the
US Tax & Legal RAG System.

Responsibilities
----------------
- Creates and manages the Groq client
- Sends prompts to the configured LLM
- Automatically falls back to the next preferred model
  if the current model errors out (e.g. rate limited / 429)
- Returns generated responses
- Logs latency and token usage
- Performs health checks

Author: Shyam Suman
Project: US Tax & Legal RAG System
"""

import logging
import random
import time
from typing import Dict, List, Optional

from groq import Groq

from config import (
    GROQ_API_KEY,
    GROQ_MODEL,
    GROQ_MODELS,
)

# ---------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# Groq Client
# ---------------------------------------------------------------------

class GroqClient:
    """
    Reusable Groq client.

    The client is created once and reused for every
    prompt generation request. If a model call fails
    (rate limit, timeout, server error, etc.), the client
    automatically retries the same request on the next
    model in `GROQ_MODELS`, in order.
    """

    def __init__(self):
        if not GROQ_API_KEY:
            raise ValueError(
                "GROQ_API_KEY not found in environment."
            )

        # Build the ordered list of models to try.
        # Falls back to a single-model list (GROQ_MODEL) if
        # GROQ_MODELS is missing/empty in config.
        self.models: List[str] = list(GROQ_MODELS) if GROQ_MODELS else [GROQ_MODEL]

        # Shuffle the fallback order once per client startup. This
        # spreads traffic (and rate-limit exposure) evenly across
        # models instead of always sending the majority of requests
        # to whichever model happens to be first in config.py. The
        # order is fixed for the lifetime of this client instance,
        # so fallback behaviour within a run stays predictable.
        random.shuffle(self.models)

        # Keep the default/primary model around for logging & reference.
        self.model = self.models[0]

        logger.info("=" * 65)
        logger.info("Initializing Groq Client")
        logger.info("Primary Model : %s", self.model)
        logger.info("Fallback Order (shuffled) : %s", self.models)

        try:
            self.client = Groq(
                api_key=GROQ_API_KEY
            )
        except Exception as error:
            logger.exception(
                "Failed to initialize Groq client."
            )
            raise RuntimeError(
                "Could not initialize Groq client."
            ) from error

        logger.info("Groq client initialized successfully.")
        logger.info("=" * 65)

    # -----------------------------------------------------------------

    # HTTP status codes that are considered PERMANENT failures.
    # Retrying these against another model wastes calls, since every
    # model will fail for the same reason (bad key, bad request, etc.).
    _PERMANENT_STATUS_CODES = {400, 401, 403, 404, 422}

    # HTTP status codes that are considered TRANSIENT and worth
    # retrying on the next model in the fallback list.
    _TRANSIENT_STATUS_CODES = {429, 500, 502, 503, 504}

    def _get_status_code(self, error: Exception) -> Optional[int]:
        """
        Best-effort extraction of an HTTP status code from an
        exception, across the different shapes the Groq SDK
        (built on `httpx`/`openai`-style clients) may raise.
        """
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code

        response = getattr(error, "response", None)
        if response is not None:
            response_status = getattr(response, "status_code", None)
            if isinstance(response_status, int):
                return response_status

        return None

    def _is_permanent_error(self, error: Exception) -> bool:
        """
        Identifies errors that will fail the SAME WAY on every model,
        so falling back is pointless: bad/expired API key, malformed
        request, prompt too long, forbidden access, etc.
        """
        status_code = self._get_status_code(error)
        if status_code is not None:
            return status_code in self._PERMANENT_STATUS_CODES

        # Fall back to matching on the error's class name / message,
        # for SDK exceptions that don't expose a status_code cleanly.
        error_name = type(error).__name__.lower()
        error_text = str(error).lower()

        permanent_markers = (
            "authenticationerror",
            "permissiondeniederror",
            "badrequesterror",
            "notfounderror",
            "unprocessableentityerror",
        )
        if any(marker in error_name for marker in permanent_markers):
            return True

        permanent_phrases = (
            "invalid api key",
            "incorrect api key",
            "unauthorized",
            "forbidden",
            "invalid request",
        )
        return any(phrase in error_text for phrase in permanent_phrases)

    def _is_transient_error(self, error: Exception) -> bool:
        """
        Identifies errors worth retrying on the next model: rate
        limits, server-side errors, timeouts, and connection issues.
        """
        status_code = self._get_status_code(error)
        if status_code is not None:
            return status_code in self._TRANSIENT_STATUS_CODES

        error_name = type(error).__name__.lower()
        error_text = str(error).lower()

        transient_markers = (
            "ratelimiterror",
            "apitimeouterror",
            "apiconnectionerror",
            "internalservererror",
            "serviceunavailableerror",
            "timeout",
            "connectionerror",
        )
        if any(marker in error_name for marker in transient_markers):
            return True

        transient_phrases = (
            "429",
            "rate limit",
            "timed out",
            "timeout",
            "connection",
            "500",
            "502",
            "503",
            "504",
        )
        return any(phrase in error_text for phrase in transient_phrases)

    # -----------------------------------------------------------------

    def _call_with_fallback(
        self,
        *,
        messages: list,
        temperature: float,
        max_completion_tokens: int,
        timeout: int,
    ):
        """
        Attempts the chat completion request across all models in
        `self.models`, in order, stopping at the first success.

        Behaviour on failure:
        - TRANSIENT errors (429 rate limit, 500/502/503/504, timeouts,
          connection errors) -> logged, and the next model is tried.
        - PERMANENT errors (400, 401, 403, 404, 422 - bad key,
          malformed request, forbidden, etc.) -> raised immediately,
          without wasting calls on the remaining models, since they
          would fail the same way.
        - Any other/unclassified error -> treated as transient (safe
          default) so a single unexpected error type doesn't abort
          the whole fallback chain.

        Raises
        ------
        RuntimeError
            If a permanent error occurs, or if every model in the
            fallback list is exhausted due to transient errors.
        """
        last_error: Optional[Exception] = None

        for index, model_name in enumerate(self.models, start=1):
            try:
                logger.info(
                    "Attempting model %d/%d : %s",
                    index,
                    len(self.models),
                    model_name,
                )

                completion = self.client.chat.completions.create(
                    model=model_name,
                    messages=messages,
                    temperature=temperature,
                    max_completion_tokens=max_completion_tokens,
                    timeout=timeout,
                )

                logger.info(
                    "Model succeeded : %s",
                    model_name,
                )

                return completion, model_name

            except Exception as error:
                last_error = error

                if self._is_permanent_error(error):
                    logger.error(
                        "Permanent error on model %s : %s | "
                        "Not retrying other models.",
                        model_name,
                        error,
                    )
                    raise RuntimeError(
                        f"Groq request failed permanently on model "
                        f"'{model_name}': {error}"
                    ) from error

                if self._is_transient_error(error):
                    logger.warning(
                        "Transient error on model %s : %s | "
                        "Trying next model...",
                        model_name,
                        error,
                    )
                else:
                    logger.warning(
                        "Unclassified error on model %s : %s | "
                        "Trying next model as a precaution...",
                        model_name,
                        error,
                    )
                continue

        logger.exception(
            "All models in fallback list failed.",
            exc_info=last_error,
        )
        raise RuntimeError(
            f"All Groq models failed. Last error: {last_error}"
        ) from last_error

    # -----------------------------------------------------------------

    def health_check(self) -> bool:
        """
        Performs a lightweight API health check.

        Sends a tiny request to verify that:
        - API key is valid
        - at least one configured model works
        - network connection works

        Automatically falls back across `self.models` if the
        first model is unavailable/rate-limited.

        Returns
        -------
        bool
            True if successful.
        """
        logger.info("=" * 65)
        logger.info("Running Groq Health Check")
        logger.info("=" * 65)

        try:
            start_time = time.perf_counter()

            completion, model_used = self._call_with_fallback(
                messages=[
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                temperature=0.0,
                max_completion_tokens=5,
                timeout=30,
            )

            print("\n================ RAW COMPLETION ================\n")
            print(completion)
            print("\n===============================================\n")

            latency = time.perf_counter() - start_time

            logger.info("Groq Health Check Passed.")
            logger.info("Model Used : %s", model_used)
            logger.info("Latency : %.2f seconds", latency)

            # Optional: log the returned text for debugging
            try:
                reply = completion.choices[0].message.content
                logger.info("Model Reply : %s", repr(reply))
            except Exception:
                logger.info("Model reply could not be extracted.")

            logger.info("=" * 65)

            return True

        except Exception:
            logger.exception("Groq Health Check Failed.")
            logger.info("=" * 65)
            return False

    # -----------------------------------------------------------------

    def generate(
        self,
        prompt_text: str,
        temperature: float = 0.0,          # deterministic by default
        max_completion_tokens: int = 4096,
        timeout: int = 120,
    ) -> Dict:
        """
        Sends a prompt to Groq and returns the generated response.

        If the first (preferred) model fails - for example it is
        rate limited (HTTP 429), times out, or errors - the request
        is automatically retried on the next model in
        `config.GROQ_MODELS`, until one succeeds or all are exhausted.

        Parameters
        ----------
        prompt_text : str
            Complete prompt generated by PromptBuilder.

        temperature : float
            Sampling temperature (default 0.0 for reproducibility).

        max_completion_tokens : int
            Maximum response tokens.

        timeout : int
            Request timeout in seconds.

        Returns
        -------
        Dict
            {
                "answer": str,
                "model": str,
                "prompt_tokens": int,
                "completion_tokens": int,
                "total_tokens": int,
                "latency_seconds": float,
            }
        """
        if not prompt_text.strip():
            raise ValueError(
                "Prompt cannot be empty."
            )

        logger.info("=" * 65)
        logger.info("Generating LLM Response")
        logger.info("Fallback Order : %s", self.models)
        logger.info(
            "Prompt Characters : %d",
            len(prompt_text),
        )
        logger.info("=" * 65)

        start_time = time.perf_counter()

        try:
            completion, model_used = self._call_with_fallback(
                messages=[
                    {
                        "role": "user",
                        "content": prompt_text,
                    }
                ],
                temperature=temperature,
                max_completion_tokens=max_completion_tokens,
                timeout=timeout,
            )

            print()

            logger.info(
                "Model Used : %s",
                model_used,
            )
            logger.info(
                "Finish Reason : %s",
                completion.choices[0].finish_reason,
            )

            latency = (
                time.perf_counter() - start_time
            )

            answer = (
                completion
                .choices[0]
                .message
                .content
                .strip()
            )

            usage = getattr(
                completion,
                "usage",
                None,
            )

            prompt_tokens = (
                getattr(
                    usage,
                    "prompt_tokens",
                    0,
                )
                if usage
                else 0
            )

            completion_tokens = (
                getattr(
                    usage,
                    "completion_tokens",
                    0,
                )
                if usage
                else 0
            )

            total_tokens = (
                getattr(
                    usage,
                    "total_tokens",
                    0,
                )
                if usage
                else 0
            )

            logger.info(
                "Generation completed successfully."
            )
            logger.info(
                "Latency : %.2f seconds",
                latency,
            )
            logger.info(
                "Prompt Tokens : %d",
                prompt_tokens,
            )
            logger.info(
                "Completion Tokens : %d",
                completion_tokens,
            )
            logger.info(
                "Total Tokens : %d",
                total_tokens,
            )
            logger.info("=" * 65)

            return {
                "answer": answer,
                "model": model_used,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "latency_seconds": round(
                    latency,
                    2,
                ),
            }

        except Exception as error:
            logger.exception(
                "Groq generation failed on all fallback models."
            )
            raise RuntimeError(
                "Failed to generate response from Groq."
            ) from error


# ---------------------------------------------------------------------
# Interactive CLI
# ---------------------------------------------------------------------

def main():
    """
    Interactive command-line interface for testing the Groq client.

    This allows independent verification of:
    - API Key
    - Model(s)
    - Network connectivity
    - Response generation
    - Automatic model fallback behaviour

    before integrating with the RAG pipeline.
    """
    try:
        groq_client = GroqClient()

        if not groq_client.health_check():
            print("\nHealth check failed.\n")
            return

        print("\n")
        print("=" * 80)
        print("US TAX & LEGAL RAG SYSTEM")
        print("Groq Client")
        print("=" * 80)
        print("Type 'exit' or 'quit' to stop.\n")

        while True:
            prompt = input("Enter prompt: ").strip()

            if prompt.lower() in {"exit", "quit"}:
                print("\nExiting...\n")
                break

            if not prompt:
                print("Prompt cannot be empty.\n")
                continue

            try:
                response = groq_client.generate(
                    prompt_text=prompt
                )

                print("\n")
                print("=" * 80)
                print("MODEL USED")
                print("=" * 80)
                print(response["model"])

                print("\n")
                print("=" * 80)
                print("ANSWER")
                print("=" * 80)
                print(response["answer"])

                print("\n")
                print("=" * 80)
                print("STATISTICS")
                print("=" * 80)
                print(
                    f"Prompt Tokens     : {response['prompt_tokens']}"
                )
                print(
                    f"Completion Tokens : {response['completion_tokens']}"
                )
                print(
                    f"Total Tokens      : {response['total_tokens']}"
                )
                print(
                    f"Latency           : {response['latency_seconds']} sec"
                )
                print("=" * 80)
                print()

            except Exception as error:
                logger.exception(error)
                print(f"\nError: {error}\n")

    except Exception as error:
        logger.exception(error)
        print("\nFailed to initialize Groq Client.\n")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    main()