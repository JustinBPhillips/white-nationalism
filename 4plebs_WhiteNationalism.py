### preprocess
#tweet-preprocessor: https://github.com/s/preprocessor
import numba
import numpy.random
import preprocessor as p
import time
import pandas as pd
import numpy as np
import re
import pysbd
from sentence_transformers import SentenceTransformer
import pickle
# PANDAS WILL DROP ROWS SILENTLY!!!  We need to set quoting=csv.QUOTE_NONE
import bz2
import csv
from tqdm import tqdm
from umap import UMAP
from hdbscan import HDBSCAN
import os
import os

homedir = os.getenv("HOME")
os.chdir(homedir+"/OneDrive/manuscripts/4chan/whiteNationalism/data/")

title = "whiteNationalism"
uniques_file = 'uniques_' + title + '.tsv'
all_file = 'all_sentences_' + title + '.tsv'
embeddings_file = 'embedding_' + title + '.pbz2'
umap_file = 'umap_' + title + '.tsv'
tsne_file = 'tsne_' + title + '.tsv'

# simple extraction from 4plebs archive (https://archive.org/details/4plebs-org-data-dump-2024-01)
# once extracted from json into tsv, run this:
# cat pol_4plebs_2024.tsv | grep -i 'white nationalism' > pol_whiteNationalism.tsv
plebs_data = pd.read_csv("./pol_whiteNationalism.tsv",sep="\t", header=None, quoting=csv.QUOTE_NONE)
plebs_data = plebs_data.rename(columns={0: "DATE", 1: "COUNTRY", 2: "TEXT"})

seg = pysbd.Segmenter(language="en", clean=False)
p.set_options(p.OPT.URL, p.OPT.MENTION, p.OPT.EMOJI, p.OPT.SMILEY)
uniqueSentences = {}
allSentences = []
indices = []
# I'm sure there are way faster ways to do this, but whatever.
# I'm also forced to str() the text, as some fields are being returned as floats by python's Pandas, I assume.
for index in tqdm(plebs_data.index):
	text = str(plebs_data['TEXT'][index]).strip()
	# remove unicode characters
	text = re.sub(r'<U\+[a-zA-Z0-9]{4,8}>', '', text)
	# remove unicode characters
	text = re.sub(r'\\[0-9]{3}', '', text)
	# specific newline character for 4chan data
	text = re.sub(r'\|>', ' ', text)
	text = re.sub(r'\|', ' ', text)
	text = re.sub(r'\>\>[0-9]{1,}', ' ', text)
	text = re.sub(r'\>[0-9]{1,}', ' ', text)
	text = p.clean(text)
	sentences = seg.segment(text)
	for sentence in sentences:
		if ("white nationalism" in sentence.lower()):
			sentence = str(sentence).strip()
			uniqueSentences.update({sentence: index})
			allSentences.append(sentence)
			indices.append(index)



npUniqueSentences = pd.DataFrame.from_dict(uniqueSentences, orient='index')
npUniqueSentences.index.name = 'Sentence'
npUniqueSentences.reset_index(inplace=True)
npUniqueSentences.to_csv(uniques_file, index=False, sep='\t', header=False, quoting=csv.QUOTE_NONE)

pd.DataFrame({'index':indices,'sentence':allSentences}).to_csv(all_file, index=False, sep='\t', quoting=csv.QUOTE_NONE)

text = pd.read_csv(uniques_file, keep_default_na=False, sep='\t', header=None,quoting=csv.QUOTE_NONE)
sentence_model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_model = sentence_model.encode(text[0], normalize_embeddings=False, show_progress_bar=True)
with bz2.BZ2File(embeddings_file, 'wb') as pkl:
	pickle.dump(embedding_model, pkl)


def send2UMAP(embedding_file,umap_file):
	with bz2.BZ2File(embedding_file, 'rb') as pkl:
		embedding_model = pickle.load(pkl)
	@numba.njit()
	def set_random_njit():
		np.random.seed(42)
	umap_model = UMAP(n_components=5, random_state=42, min_dist=0.0, spread=0.5, metric='cosine', verbose=True)
	set_random_njit()
	umap_model.fit(embedding_model)
	umap_embedding = umap_model.embedding_
	pd.DataFrame(umap_embedding).to_csv(umap_file, sep="\t", index=False, header=False)

def send2OpenTSNE(embedding_file,tsne_file):
	import openTSNE
	import csv
	umapped = pd.read_csv(embedding_file, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)
	dims = np.array(umapped)
	# https://www-nature-com.ezproxy.waikato.ac.nz/articles/s41467-019-13056-x
	if len(dims)>10000:
		# we must downsample to make init computationally feasible
		np.random.seed(42)
		shuffle = np.random.permutation(list(range(dims.shape[0])))
		reverse = np.argsort(shuffle)
		dims_shuffle = dims[shuffle]
		print("Downsampling.  Confirming unshuffle function works... " + str(np.array_equal(dims,dims_shuffle[reverse])))
		shuffle_sample = dims_shuffle[:25000]
		shuffle_rest = dims_shuffle[25000:]
		high_perplex = len(shuffle_sample) / 100
		print("30 and " + str(high_perplex))
		affinities_multiscale_mixture = openTSNE.affinity.Multiscale(shuffle_sample, perplexities=[30, high_perplex],
		                                                             metric="cosine", n_jobs=-1, random_state=42,
		                                                             verbose=True)
		init = openTSNE.initialization.pca(shuffle_sample, random_state=42, verbose=True)
		tsne_model = openTSNE.TSNE(learning_rate=len(shuffle_sample) / 12, n_jobs=8, verbose=True)
		sample_embedding_multiscale = tsne_model.fit(affinities=affinities_multiscale_mixture, initialization=init)
		rest_embedding_multiscale = sample_embedding_multiscale.prepare_partial(shuffle_rest)
		embedding_multiscale = np.vstack((sample_embedding_multiscale, rest_embedding_multiscale))[reverse]
	else:
		high_perplex = len(dims) / 100
		print("30 and " + str(high_perplex))
		affinities_multiscale_mixture = openTSNE.affinity.Multiscale(dims, perplexities=[30, (high_perplex)],metric="cosine", n_jobs=-1, random_state=42,verbose=True)
		init = openTSNE.initialization.pca(dims, random_state=42, verbose=True)
		print("Learning rate: " + str(len(dims) / 12))
		tsne_model = openTSNE.TSNE(learning_rate=len(dims) / 12, n_jobs=8, verbose=True)
		embedding_multiscale = tsne_model.fit(affinities=affinities_multiscale_mixture, initialization=init)
	pd.DataFrame(embedding_multiscale).to_csv(tsne_file, header=False, index=False, sep='\t',quoting=csv.QUOTE_NONE)

def send2HDBSCAN(umap_file,cluster_file,min_cluster_percentage,min_samples):
	import csv
	umapped = pd.read_csv(umap_file,keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)
	dims = np.array(umapped)
	min_cluster_size = round(len(dims)*min_cluster_percentage)
	print("Min cluster size: " + str(min_cluster_size))
	hdbscan_model = HDBSCAN(min_samples=min_samples,min_cluster_size=min_cluster_size, cluster_selection_method='leaf', prediction_data=False)
	hdbscan_model.fit(dims)
	clusters = hdbscan_model.labels_
	#probs = hdbscan_model.probabilities_
	uniques = np.unique(np.array(clusters), return_counts=True)
	print("Distribution: " + str(uniques))
	print("Uniques: " + str(len(uniques[0])))
	print("-1 %: " + str(uniques[1][0]/len(clusters)))
	# convert array into dataframe
	#pd.DataFrame({'clusters': clusters, 'probs': probs}).to_csv(cluster_file, index=False, sep='\t',quoting=csv.QUOTE_NONE)
	pd.DataFrame({'clusters': clusters}).to_csv(cluster_file, index=False, sep='\t',quoting=csv.QUOTE_NONE)

def send2BERTopic(uniques, embeddings, clusters):
	from bertopic import BERTopic
	from sklearn.feature_extraction.text import CountVectorizer
	import spacy
	# spacy.cli.download('en_core_web_sm')
	spacy.load('en_core_web_sm')
	from bertopic.representation import PartOfSpeech
	from bertopic.vectorizers import ClassTfidfTransformer
	from sentence_transformers import SentenceTransformer
	from bertopic.dimensionality import BaseDimensionalityReduction
	from bertopic.cluster import BaseCluster
	from bertopic.representation import KeyBERTInspired
	from bertopic.vectorizers import OnlineCountVectorizer
	from bertopic.representation import MaximalMarginalRelevance
	uniques = pd.read_csv(uniques, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE)[0]
	embeddings = np.array(pd.read_csv(embeddings, keep_default_na=False, sep='\t', header=None, quoting=csv.QUOTE_NONE))
	clusters = pd.read_csv(clusters, keep_default_na=False, sep='\t', quoting=csv.QUOTE_NONE)['clusters']
	representation_models = [MaximalMarginalRelevance(diversity=1),PartOfSpeech("en_core_web_sm")]
	vectorizer_model = OnlineCountVectorizer(stop_words="english")
	ctfidf_model = ClassTfidfTransformer(bm25_weighting=True, reduce_frequent_words=True)
	embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
	empty_reduction_model = BaseDimensionalityReduction()
	empty_cluster_model = BaseCluster()
	# Fit BERTopic without constructing embeddings
	topic_model = BERTopic(
		embedding_model=embedding_model,
		umap_model=empty_reduction_model,
		hdbscan_model=empty_cluster_model,
		vectorizer_model=vectorizer_model,
		ctfidf_model=ctfidf_model,
		representation_model=representation_models,
		verbose=True
	).fit(uniques, embeddings=embeddings, y=clusters)
	print("Customizing representative docs.")
	#representative_docs is screwed up for a number of reasons (pysbd doesn't properly parse " '; it tends to give long answers, etc.) let's fix that:
	revised_docs = pd.DataFrame({'Document': uniques, 'Topic': topic_model.topics_})
	revised_docs = revised_docs[~revised_docs.Document.str.contains("\"")]
	revised_docs = revised_docs[~revised_docs.Document.str.contains("\'")]
	revised_docs = revised_docs[revised_docs.Document.str.len() > 15]
	revised_docs = revised_docs[revised_docs.Document.str.len() < 225]
	revised_docs = revised_docs.reset_index(drop=True)
	# sample all of the remaining data and give 10 sentences.
	tmprepr_docs_mappings, tmprepr_docs, tmprepr_docs_indices, tmprepr_docs_ids = topic_model._extract_representative_docs(c_tf_idf=topic_model.c_tf_idf_,
	                                                      documents=revised_docs,
	                                                      topics=topic_model.topic_representations_,
	                                                      nr_samples=round(revised_docs.shape[0]),
	                                                      nr_repr_docs=10)
	topic_model.representative_docs_= tmprepr_docs_mappings
	return topic_model

def resultsFromBERT2HDBSCANtopics(topic_model,hdbscan_file):
	import re
	map_BERTandHDBSCAN_topic_numbers = pd.DataFrame({'BERT': np.array(topic_model.topics_),
	                                                 'HDBSCAN': np.array(pd.read_csv(hdbscan_file,
	                                                                                 keep_default_na=False, sep='\t',
	                                                                                 quoting=csv.QUOTE_NONE)[
		                                                                     'clusters'])}).drop_duplicates().set_index(
	'BERT')['HDBSCAN'].to_dict()
	results = topic_model.get_topic_info()
	# let's repopulate representative docs...
	for i in range(len(results['Topic'])):
		results['Topic'][i] = map_BERTandHDBSCAN_topic_numbers.get(results['Topic'][i])
		results['Name'][i] = re.sub("^[0-9]*_",str(results['Topic'][i])+"_",results['Name'][i])
	return results



def results2browser(results):
	import os, tempfile
	resultsHTML = pd.DataFrame(results).to_html(classes=["table-bordered", "table-striped", "table-hover"])
	tmp = tempfile.NamedTemporaryFile(delete=False,suffix=".html")
	try:
		print(tmp.name)
		tmp.write(resultsHTML.encode('ascii'))
		os.system("firefox " + tmp.name)
		time.sleep(2)
		tmp.close()
	finally:
		os.unlink(tmp.name)
	return

def results2csv(results,results_file):
	results.to_csv(results_file,index=False, sep='\t')



send2UMAP(embeddings_file,umap_file)
send2OpenTSNE(umap_file,tsne_file)

min_cluster_percentage = 0.005
min_samples = 2
hdbscan_file = "hdbscan_" + title + "_"+str(min_cluster_percentage)+"_mcp_"+str(min_samples)+"_ms.tsv"
send2HDBSCAN(tsne_file,hdbscan_file,min_cluster_percentage,min_samples)
model_BERTopic = send2BERTopic(uniques_file,tsne_file,hdbscan_file)
results2browser(resultsFromBERT2HDBSCANtopics(model_BERTopic,hdbscan_file))
results2csv(resultsFromBERT2HDBSCANtopics(model_BERTopic,hdbscan_file),"results_"+hdbscan_file)
