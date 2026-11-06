"""Lazy, in-process version of the original free Qwen GGUF backend."""

import os


class LocalInferenceError(RuntimeError):
    """A local model could not load or execute, rather than malformed prose."""


def _available_threads():
    # Respect runner/container CPU affinity rather than oversubscribing a
    # two-core runner with the historical four-thread setting.
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


class LocalLLM:
    def __init__(self, max_input_tokens=2048):
        if (isinstance(max_input_tokens, bool) or not isinstance(max_input_tokens, int)
                or not 1 <= max_input_tokens <= 4488):
            raise ValueError('llm.local.max_input_tokens must be an integer between 1 and 4488.')
        self.max_input_tokens = max_input_tokens
        self._model = None

    def generate(self, messages):
        try:
            return self._generate(messages)
        except Exception as exc:
            # Preserve the traceback for safe type/location diagnostics, but
            # never put provider messages or prompts in the public exception.
            raise LocalInferenceError("Local model initialization or inference failed.") from exc

    def _generate(self, messages):
        if self._model is None:
            from llama_cpp import Llama
            self._model = Llama.from_pretrained(
                repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF',
                filename='qwen2.5-3b-instruct-q4_k_m.gguf',
                n_ctx=5000,
                n_threads=_available_threads(),
                verbose=False,
            )
        # Count using the actual model tokenizer. Leave room for chat-template
        # tokens and output; upstream's 16384 API token limit cannot fit here.
        # Bound CPU prefill cost. Prompts put title and abstract before full
        # text, so truncation removes the later paper text first.
        remaining = self.max_input_tokens
        truncated = []
        for message in messages:
            tokens = self._model.tokenize(message['content'].encode('utf-8'), add_bos=False)
            tokens = tokens[:remaining]
            remaining -= len(tokens)
            truncated.append({**message, 'content': self._model.detokenize(tokens).decode('utf-8', errors='ignore')})
        response = self._model.create_chat_completion(
            messages=truncated, temperature=0, max_tokens=256,
        )
        return response['choices'][0]['message']['content']
