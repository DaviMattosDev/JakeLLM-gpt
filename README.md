# 🐶 JakeLLM — Um LLM Multimodal Caseiro em Homenagem ao Jake

<img src="https://github.com/user-attachments/assets/00b83bbf-0695-490e-8380-86ba16254be3" alt="JakeLLM" height="400px" width="900px"/> 



**JakeLLM** é um projeto de modelo de linguagem inspirado nos grandes LLMs como GPT, DeepSeek e LLaMA, mas construído de forma simples, com recursos limitados e muito coração 💛 — em homenagem ao meu pinscher Jake.

---

## ✨ Objetivo

Criar um modelo de linguagem **do zero**, em português, treinado localmente ou no Google Colab, com capacidade de inferência textual e modularidade para crescimento futuro (multimodalidade, geração de imagens, embeddings, etc).

---

## 📂 Estrutura do Projeto

```
JakeLLM/
├── model/
│   └── gpt_model.py # Estrutura do modelo Transformer
│   └── train.py  # Script de treinamento   
├── tokenizer/
│   ├── train_tokenizer.py # Treinamento do tokenizer BPE
│   └── vocab.json, merges.txt
├── data/raw
│   └── corpus.txt         # Textos para o tokenizer
│            
├── requirements.txt       # Dependências do projeto
└── README.md              # Este arquivo
```

---

## 🧠 Modelo

O modelo implementa um mini-GPT com:
- 🔢 **Embedding**: Embeddings de token e posição
- 🔁 **Camadas**: 4 camadas de Transformer com atenção multi-cabeça
- 📏 **Parâmetros ajustáveis**: Tamanho do vocabulário, `dim`, número de `heads`, `block_size`

---

## 🗣️ Tokenizer

Usamos o `ByteLevelBPETokenizer` da HuggingFace, treinado com um corpus em português contendo:
- Artigos da Wikipédia
- Textos públicos
- Notas de código

Treinamento com:
```bash
python tokenizer/train_tokenizer.py
```

---

## 🏋️ Treinamento

Script principal: `train.py`

```bash
python train.py
```

O modelo treina em pequenos lotes e imprime o `loss` por época.

### Exemplo de resultado:

```
Epoch 1 - Loss: 2.3912
Epoch 2 - Loss: 1.8473
Epoch 3 - Loss: 1.3521
```

---

## 💻 Requisitos

```bash
pip install -r requirements.txt
```

### Principais bibliotecas:
- `torch`
- `transformers`
- `datasets`
- `tokenizers`

---

## 📊 Exemplo de Execução

```python
>>> from model.gpt_model import GPT
>>> model = GPT(...)
>>> prompt = "Qual é a capital do Brasil?"
>>> model.generate(prompt)
"Brasília é a capital do Brasil."
```

---

## 🖼️ Homenagem ao Jake

O projeto leva o nome **JakeLLM** em homenagem ao meu cachorro Jake, que sempre esteve ao meu lado nos momentos mais difíceis. 🐾

---

## 📌 Status do Projeto

- ✅ Tokenizer funcional
- ✅ Modelo Transformer básico
- ✅ Treinamento local
- 🔜 Inference em tempo real
- 🔜 Suporte a geração multimodal
- 🔜 Interface web com Gradio

---

## 🚀 Meta

Mostrar que é possível fazer muito mesmo com poucos recursos — basta vontade, estratégia e paixão por IA.  
Este projeto é a base de algo muito maior: **um LLM brasileiro, pessoal e original**.

---

## 📬 Contato

Feito com ❤️ por [Davi Mattos](https://github.com/DaviMattosDev)
