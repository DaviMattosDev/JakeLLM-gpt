from tokenizers import ByteLevelBPETokenizer

tokenizer = ByteLevelBPETokenizer()
tokenizer.train(files=[r"./data/raw/corpus.txt"], vocab_size=32000, min_frequency=2)
tokenizer.save_model("tokenizer")