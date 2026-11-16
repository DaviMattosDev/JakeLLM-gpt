# 🐶 JakeLLM — Um mini-GPT em português, em homenagem ao Jake

JakeLLM é um projeto educacional de modelo de linguagem treinado do zero, dedicado ao pinscher Jake. Implementa geração de texto; multimodalidade e interface web são ideias para o futuro.

## Instalação

Execute os comandos a partir da raiz do repositório, com Python 3.10 ou superior:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Para Windows, ative o ambiente com `.venv\Scripts\activate`. Para uma instalação de PyTorch com CUDA, use a distribuição apropriada ao seu ambiente.

## Tokenizer

```bash
python -m tokenizer.train_tokenizer
```

O corpus incluído tem aproximadamente 4,4 KB e serve apenas para testar o fluxo. Para treinamento real, use um corpus maior, limpo, representativo do português desejado e com permissão de uso. Treine o tokenizer somente nos dados de treinamento, sem usar a validação:

```bash
python -m tokenizer.train_tokenizer --files dados/treino.txt --vocab-size 8000
```

Isso gera `tokenizer/vocab.json` e `tokenizer/merges.txt`. Preserve esses arquivos junto ao checkpoint: alterar o tokenizer exige treinar um novo modelo.

## Treinamento

Por padrão, o script usa até 5.000 artigos da Wikipédia em português, obtidos pela biblioteca `datasets` (requer internet):

```bash
python -m model.train
```

Também é possível treinar inteiramente com arquivos locais UTF-8, com um documento por linha não vazia:

```bash
python -m model.train --train-file dados/treino.txt --validation-file dados/validacao.txt
```

Sem `--validation-file`, 10% dos documentos são reservados para validação antes da divisão em sequências. Cada documento deve conter mais tokens que `--block-size` para produzir uma sequência; documentos curtos e restos incompletos são descartados. O tokenizer e os arquivos de treino/validação devem ser preparados sem sobreposição de dados.

Exemplo menor para CPU:

```bash
python -m model.train --train-file data/raw/corpus.txt --dim 64 --layers 2 --heads 4 --block-size 64 --epochs 3
```

Esse exemplo verifica o funcionamento; não produz um assistente útil. Use `python -m model.train --help` para ajustar os parâmetros.

O treinamento usa atenção causal, dropout, AdamW e clipping de gradientes. Cada época imprime loss de treino, loss de validação e perplexidade. O melhor checkpoint de validação é salvo em `model_weights.pth`, incluindo configuração, identificação do tokenizer e época. A seed padrão é 42; isso não garante resultados idênticos em todo hardware. Não há retomada do estado do otimizador.

## Geração

Depois de treinar:

```bash
python -m inference.cli --temperature 0.8 --top-k 20 --max-new-tokens 50
```

Digite um texto inicial ou `sair` para encerrar. O modelo gera uma continuação; não foi treinado para seguir instruções ou responder perguntas com precisão. A geração mantém apenas os últimos `block_size` tokens como contexto.

Checkpoints antigos com a arquitetura padrão ainda podem ser carregados, mas precisam ser retreinados: perdas do treinamento anterior, sem máscara causal, não são comparáveis às novas perdas.

## Estrutura

- `model/gpt_model.py`: Transformer causal, com tamanho de contexto configurável.
- `model/train.py`: treinamento e validação por documento.
- `tokenizer/train_tokenizer.py`: treinamento do tokenizer byte-level BPE.
- `inference/cli.py`: geração interativa com amostragem top-k.
- `tests/test_model.py`: regressões de causalidade, aprendizado, contexto e checkpoints.

## Testes

```bash
python -m unittest discover -s tests -v
```

## Como melhorar a qualidade

1. Amplie e limpe o corpus; remova duplicatas e mantenha validação independente.
2. Treine o tokenizer no corpus representativo e retreine o modelo com atenção causal.
3. Compare loss de validação e amostras com prompts fixos antes de aumentar parâmetros ou épocas.
4. Para comportamento de assistente, adicione posteriormente ajuste supervisionado com exemplos de instrução e resposta. O treino atual é de continuação de texto.

O projeto é pequeno e educacional. Treinar do zero com poucos dados não oferece a capacidade de modelos pré-treinados de grande escala.

## Homenagem

O projeto leva o nome JakeLLM em homenagem ao cachorro Jake, companheiro de Davi Mattos. 🐾

Criado por [Davi Mattos](https://github.com/DaviMattosDev). Licença MIT.
