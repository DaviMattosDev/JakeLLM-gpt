import torch
from torch.utils.data import Dataset, DataLoader
from model.gpt_model import GPT
from tokenizers import ByteLevelBPETokenizer
from datasets import load_dataset
import torch.nn.functional as F
from tqdm import tqdm  

class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, block_size=128):
        self.examples = []
        for txt in texts:
            encoded = tokenizer.encode(txt)
            tokens = encoded.ids
            for i in range(0, len(tokens) - block_size, block_size):
                self.examples.append(torch.tensor(tokens[i:i+block_size]))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]

# Carregar dataset / Wikipedia em português / 1% do dataset / 5000 textos 
print("🔄 Carregando dataset da Wikipedia em português...")
dataset = load_dataset("wikimedia/wikipedia", "20231101.pt", split="train[:1%]")
texts = dataset["text"][:5000]

# Carregar tokenizer treinado
print("🔠 Carregando tokenizer...")
tokenizer = ByteLevelBPETokenizer("tokenizer/vocab.json", "tokenizer/merges.txt")

# Preparar dados
print("🧹 Preparando dados...")
data = TextDataset(texts, tokenizer, block_size=128)
dataloader = DataLoader(data, batch_size=4, shuffle=True)

# Criar modelo
print("🧠 Inicializando modelo...")
model = GPT(vocab_size=tokenizer.get_vocab_size(), dim=256, layers=4, heads=4)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Treinar
print("🚀 Iniciando treinamento...\n")
for epoch in range(1, 4):  # 3 epochs
    total_loss = 0
    steps = 0
    progress_bar = tqdm(dataloader, desc=f"🧪 Epoch {epoch}")
    
    for batch in progress_bar:
        batch = batch.to(device)
        outputs = model(batch[:, :-1])
        loss = F.cross_entropy(outputs.view(-1, outputs.size(-1)), batch[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        steps += 1
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / steps
    print(f"📉 Epoch {epoch} finalizada — Loss média: {avg_loss:.4f}\n")

# Salvar modelo
print("💾 Salvando pesos finais...")
torch.save(model.state_dict(), "model_weights.pth")
print("✅ Treinamento concluído com sucesso!")
