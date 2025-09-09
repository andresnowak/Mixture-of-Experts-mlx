from mlx_lm.utils import load
from mlx_lm.generate import generate

model, tokenizer = load("/Users/andresnowak/.lmstudio/models/mlx-community/gemma-3-12b-it-qat-4bit")


prompt = "hello"

if tokenizer.chat_template is not None:
    messages = [{"role": "user", "content": prompt}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

response = generate(model, tokenizer, prompt=prompt, verbose=True)
