"""Lazy, in-process version of the original free Qwen GGUF backend."""


class LocalLLM:
    def __init__(self):
        self._model = None

    def generate(self, messages):
        if self._model is None:
            from llama_cpp import Llama
            self._model = Llama.from_pretrained(
                repo_id='Qwen/Qwen2.5-3B-Instruct-GGUF',
                filename='qwen2.5-3b-instruct-q4_k_m.gguf',
                n_ctx=5000,
                n_threads=4,
                verbose=False,
            )
        # Count using the actual model tokenizer. Leave room for chat-template
        # tokens and output; upstream's 16384 API token limit cannot fit here.
        remaining = 5000 - 768 - 256
        truncated = []
        for message in messages:
            tokens = self._model.tokenize(message['content'].encode('utf-8'), add_bos=False)
            tokens = tokens[:remaining]
            remaining -= len(tokens)
            truncated.append({**message, 'content': self._model.detokenize(tokens).decode('utf-8', errors='ignore')})
        response = self._model.create_chat_completion(
            messages=truncated, temperature=0, max_tokens=768,
        )
        return response['choices'][0]['message']['content']
