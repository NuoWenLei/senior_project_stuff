from Similar_Word_Scraper import Similar_Word_Scraper
from Embed_Generator import Embed_Generator
import json, numpy as np

def main():
	with open("../params.json", "r") as params_json:
		params = json.load(params_json)

	Similar_Word_Scraper.initialize_nltk()

	word_scraper = Similar_Word_Scraper(params["STARTING_VOCAB"], params["VOCAB_DEPTH"])

	words = word_scraper()

	embed_generator = Embed_Generator(words)

	embeds = embed_generator()

	with open(params["PATH_TO_VOCAB"], "w") as vocab_json:
		json.dump(words, vocab_json)
	
	with open(params["PATH_TO_EMBEDDING"], "wb") as embed_npy:
		np.save(embed_npy, embeds)


if __name__ == "__main__":
	main()



