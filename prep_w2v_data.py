'''
	Identify multi-term UMLS concepts from MEDLINE texts and PMCOA full texts. 
	Split texts into sentences for gensim.

	Created on Oct 6, 2016
	@author Wei Wei
'''

## refer to deep-qa-master/learn_embeddings.py
## raw MEDLINE data: idash-data:~/data2/medline_docs_by_year
## processed MEDLINE: idash-data:~/data/gensim
## raw PMCOA data: ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/oa_bulk/

## trained model: idash-cloud:/home/w2wei/projects/word2vec/models


## before installing metamap, test the speed of metamap on a 32 cpu vm, estimate the time cost. if it takes too long, restrict resources. 
## run metamap, record umls concepts for every medline doc. analyze every medline doc, concatenate every umls concept using underscores. 