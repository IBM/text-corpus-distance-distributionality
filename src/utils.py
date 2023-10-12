import json
import os
import numpy as np
import pandas as pd
from compcor.text_tokenizer_embedder import STTokenizerEmbedder
from sklearn.feature_extraction.text import TfidfTransformer

QUORA = r'quora_duplicates.json'

def sort_numsuffix(x):
	# sort by suffix length if has numeric suffix after _ else by alphabetical
	# e.g. ['DC_10', 'DC_5', 'PR', 'A'] should be sorted ['A', 'DC_5', 'DC_10', 'PR']
	xs = str(x).split("_")
	return (xs[0],) if len(xs) == 1 else (xs[0], float(xs[1]))


def load_paraphrases(filename='quora_duplicates.csv', subsample=None):
	# returns two lists (corpora) of paraphrases
	# if subsample is not None, provide an integer to sample instances
	fpath = os.path.join(os.path.dirname(__file__), '../data', filename)
	df = pd.read_csv(fpath)
	colnames = df.columns
	if subsample is not None:
		subsample = max(2, min(int(subsample), len(df)))
		df = df.sample(n=subsample, replace=False)
	return df[colnames[0]].to_list(), df[colnames[1]].to_list()


# def load_paraphrases(filename, subsample=None):
# 	# element in the json file is a pair of paraphrases
# 	# returns two lists (corpora) of paraphrases
# 	# if subsample is not None, provide an integer to sample instances
# 
# 	fpath = os.path.join(os.path.dirname(__file__), '../data', filename)
# 	with open(fpath, 'rb') as fp:
# 		pp = json.load(fp)
# 		c1, c2 = [pair[0] for pair in pp], [pair[1] for pair in pp]
# 		sz = len(c1)
# 
# 	if subsample is not None:
# 		subsample = max(2, min(int(subsample), sz))
# 		if subsample < sz:
# 			# return only sub-sampled indices
# 			idx = np.random.choice(a=sz, size=subsample, replace=False)
# 			return [c1[ii] for ii in idx], [c2[ii] for ii in idx]
# 		else:
# 			return c1, c2
# 	else:
# 		return c1, c2


def common_tokens_matrix(corpus1, corpus2, tokenizer=STTokenizerEmbedder(), top=5000, idf=True):
	from more_itertools import flatten
	from collections import Counter

	tokens_vec1 = tokenizer.tokenize_sentences(corpus1)
	tokens_vec2 = tokenizer.tokenize_sentences(corpus2)
	# Counter in each
	tokens_vec_ctr1 = [Counter(vec) for vec in tokens_vec1]
	tokens_vec_ctr2 = [Counter(vec) for vec in tokens_vec2]

	flat_tokens_vec1 = list(flatten(tokens_vec1))
	flat_tokens_vec2 = list(flatten(tokens_vec2))

	common_words = set([word for word, freq in Counter(flat_tokens_vec1 + flat_tokens_vec2).most_common(top)])

	matrix1 = np.array([[ctr[token] for token in common_words] for ctr in tokens_vec_ctr1])
	matrix2 = np.array([[ctr[token] for token in common_words] for ctr in tokens_vec_ctr2])

	if idf:
		# transform with IDF
		tfidf = TfidfTransformer(use_idf=True)
		matrix1 = tfidf.fit_transform(X=matrix1)
		matrix2 = tfidf.fit_transform(X=matrix2)

	return matrix1, matrix2