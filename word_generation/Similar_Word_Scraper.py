from nltk.corpus import wordnet as wn, stopwords
from nltk import word_tokenize
from tqdm import tqdm
import nltk, string

class Similar_Word_Scraper():

	def __init__(self, starting_word, search_depth, allowed_pos = ["NN", "NNP", "NNS", "NNPS", "JJ", "JJR", "JJS"]):
		self.stop = stopwords.words("english")
		self.starting_word = starting_word
		self.depth = search_depth
		self.allowed_pos = allowed_pos


	
	def initialize_nltk():
		for m in ["averaged_perceptron_tagger", "stopwords", "wordnet", "punkt"]:
			nltk.download(m)



	# Limit part of speech of words recorded
	def is_adj_or_noun(self, pos):
		if pos in self.allowed_pos:
			return True
		return False




	def queue_all_words_through_wordnet(self, word, curr_set, queue, old_queue):

		# Search through all synsets of word
		for ss in wn.synsets(word):

			# Record every new word that appears in the definition
			for w,p in nltk.pos_tag(word_tokenize(ss.definition().lower())):

				# Strict filters to prevent higher time complexity and strengthen data quality
				if (w not in string.punctuation) and (w not in self.stops) and (w not in [word for word,_ in curr_set]) and self.is_adj_or_noun(p) and (w not in queue) and (w not in old_queue):
					queue.append(w)

		return queue





	# In order to keep track of the depth of any given word,
	# vocab generation is a 2-function process
	def vocab_generation(self):
		# Initialize word set and queue
		curr_set = []
		queue = [self.starting_word]

		# Breadth-first search (BFS) down the tree
		for i in range(self.depth):
			print(i)

		# Create queue for all words in the next depth
		new_queue = []

		for w in tqdm(queue):
			# Record current word with current depth
			curr_set.append([w, i])

			# Queue all direct children nodes of this word
			new_queue = self.queue_all_words_through_wordnet(w, curr_set, new_queue, queue)

		print(f"Length of new queue: {len(new_queue)}")

		# Overwrite old queue
		queue = new_queue.copy()

		return curr_set



	def call(self):
		return self.vocab_generation()
		
