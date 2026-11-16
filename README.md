# 🐶 JakeLLM — um mini-GPT experimental em português

Projeto de pesquisa/estudo inspirado na arquitetura GPT e criado em homenagem ao Jake. Implementa um **Transformer decoder-only com atenção causal**, tokenizer BPE de bytes, treinamento de previsão do próximo token, geração de texto e API local opcional.

> **Importante:** este repositório contém **código, não um modelo pré-treinado**. O corpus de exemplo é minúsculo e serve apenas para validar o pipeline. Para obter respostas úteis em português, será preciso um conjunto de dados substancialmente maior, treino adequado e avaliação rigorosa. Não há multimodalidade nem desempenho comparável aos grandes LLMs atuais.

## Início rápido (Python 3.10+)

Execute os comandos **a partir da raiz do repositório**:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m tokenizer.train_tokenizer
python -m model.train --epochs 3
python -m inference.cli
```

Para uma demonstração rápida em CPU:

```bash
python -m model.train --dim 64 --layers 2 --heads 4 --block-size 64 --epochs 1 --max-steps 5
python -m inference.cli --prompt "Olá, Jake" --max-new-tokens 20 --temperature 0.8
```

**Atenção:** o treinamento rápido substitui o checkpoint de mesmo caminho. Após treinar com outra arquitetura, use os pesos e tokenizer correspondentes.

## Treinamento e dados

- `tokenizer/train_tokenizer.py` treina um BPE a partir de `data/raw/corpus.txt`, com tokens especiais `<pad>`, `<bos>`, `<eos>` e `<unk>`.
- `model/train.py` lê esse corpus por padrão. O dataset cria pares deslocados de um token e separa validação **antes** de construir janelas. Há clipping de gradientes, avaliação a cada época e persistência do melhor checkpoint.
- Para experimentar a amostra de Wikipédia em português, use `python -m model.train --wikipedia` (requer internet e disponibilidade do dataset). O tokenizer deve ser treinado previamente em um corpus representativo dessa fonte.
- Os pesos são salvos em `checkpoints/jakellm.pt` junto aos hiperparâmetros. Tokenizer e pesos gerados são ignorados pelo Git; guarde ambos com cuidado para reproduzir uma execução.
- Alterações na arquitetura causal invalidam pesos obtidos com a implementação antiga. **Retreine o modelo.**

Opções úteis:

```bash
python -m tokenizer.train_tokenizer data/raw/corpus.txt --vocab-size 8000
python -m model.train --help
python -m inference.cli --help
```

## API HTTP local (opcional)

```bash
python -m pip install -r requirements-api.txt
uvicorn api.main:app --host 127.0.0.1 --port 8000
```

- `GET /` abre a interface web básica no navegador.
- `GET /health` verifica se o processo HTTP está ativo; não indica que há pesos carregados.
- `GET /ready` verifica se checkpoint e tokenizer podem ser carregados.
- `POST /generate` aceita `prompt`, `max_new_tokens`, `temperature`, `top_k` e devolve `{"text": "..."}`.

```bash
curl -X POST http://127.0.0.1:8000/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"Olá!","max_new_tokens":20}'
```

Variáveis opcionais: `JAKELLM_CHECKPOINT`, `JAKELLM_TOKENIZER_DIR`, `JAKELLM_DEVICE` (`auto`, `cpu`, `cuda`).

**Segurança:** esta API não implementa autenticação, rate limit, quotas nem streaming. Use apenas localmente, vinculada ao loopback; **não publique na internet** sem controles adicionais.

## Testes

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

A suíte verifica causalidade (tokens futuros não alteram previsões anteriores), limites de contexto, geração, janelas de treino e validações da API. GitHub Actions executa os testes em push e pull requests.

## Estrutura

```text
api/main.py                 # API local opcional
inference/cli.py            # CLI de chat
inference/generate.py       # carregamento seguro e geração
model/gpt_model.py          # Transformer causal
model/train.py              # treino e validação
tokenizer/train_tokenizer.py # treinamento do tokenizer
data/raw/corpus.txt         # corpus de demonstração
tests/                      # testes de regressão
```

## Limitações e próximos passos

O JakeLLM é um **projeto de pesquisa**, não um assistente pronto. Um chatbot de qualidade requer dados de treinamento e instrução licenciados, avaliações, recursos de hardware e um modelo adequado ao objetivo. Evoluções possíveis incluem cache KV, fine-tuning instrucional, streaming, benchmarks comparativos, autenticação e uma UI web. A arquitetura aqui é somente textual; multimodalidade permanece como objetivo futuro.

Licença: MIT. Projeto original de [Davi Mattos](https://github.com/DaviMattosDev).
