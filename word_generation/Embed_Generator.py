import numpy as np, spacy
from tqdm import tqdm

class Embed_Generator():
	
	def __init__(self, vocab, spacy_module = "en_core_web_md"):
		self.vocab = vocab
		self.nlp = spacy.load(spacy_module)

	def __call__(self):
		mat = []
		for w in tqdm(self.vocab):
			mat.append(self.nlp(w).vector)
		
		return np.array(mat)

