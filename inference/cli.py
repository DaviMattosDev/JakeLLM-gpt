
import sys
import os
import torch
from tokenizers import ByteLevelBPETokenizer
import torch.nn.functional as F

# Adiciona o diretório raiz ao sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from model.gpt_model import GPT

# Carrega o tokenizer
tokenizer = ByteLevelBPETokenizer.from_file("tokenizer/vocab.json", "tokenizer/merges.txt")

# Inicializa o modelo com o vocabulário correto
vocab_size = tokenizer.get_vocab_size()
model = GPT(vocab_size=vocab_size)
model.load_state_dict(torch.load("model_weights.pth", map_location="cpu"))
model.eval()

def generate(model, tokenizer, prompt, max_len=50, temperature=1.0, top_k=20):
    model.eval()
    tokens = tokenizer.encode(prompt).ids
    input_tensor = torch.tensor(tokens).unsqueeze(0)

    for _ in range(max_len):
        with torch.no_grad():
            output = model(input_tensor)
        logits = output[0, -1, :] / temperature

        # Top-k filtering
        top_k = min(top_k, logits.size(-1))
        values, indices = torch.topk(logits, top_k)
        probs = F.softmax(values, dim=-1)

        next_token = indices[torch.multinomial(probs, num_samples=1)].item()
        tokens.append(next_token)
        input_tensor = torch.tensor(tokens).unsqueeze(0)

    return tokenizer.decode(tokens)

# CLI interativo
print("JakeLLM pronto! Digite 'sair' para encerrar.")
while True:
    prompt = input("Você: ")
    if prompt.lower() in ["sair", "exit", "quit"]:
        break
    response = generate(model, tokenizer, prompt)
    print("JakeLLM:", response)
