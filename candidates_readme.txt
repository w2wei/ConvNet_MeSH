The corpus_medline module retrieves MEDLINE documents from PubMed, selects the latest 1M MEDLINE (end by Dec 2015) documents to train Doc2Vec models, 
processes and indexes titles and abstracts of 9M MEDLINE documents, retrieves documents using indexed 9M documents (title and abstract) using BM25 algorithm, 
analyze BM25 performance in terms of MeSH coverage, learns distributed Word2Vec and Doc2Vec representations on the 1M documents using gensim, 
computes the inversed document frequencies of terms in the vocabulary,  
represents documents using a Doc2Vec+TFIDF representation and finds KNN documents using cosine similarity based on this representation.


P for production, A for analysis, number indicates the order of scripts to be executed

P1.candidates_build.py (old name: corpus_medline_build.py)
   Retrieve all PMID for given years from PubMed, and then download MEDLINE associated with these PMID. 

P2.candidates_sort.py (old name: corpus_medline_sort.py)
   Find the latest 9M MEDLINE from 10M records during 2004 and 2016 because most papers after Dec 2015 do not have MeSH terms. This class will be deprecated.

P3.candidates_index.py (old name: corpus_medline_index.py)
   Process the 9M raw MEDLINE docuemnts and extract titles and abstracts, and index them using pyLucene.

P4.candidates_retrieve.py (old name: corpus_medline_retrieve.py)
   Retrieve documents and rank them using BM25 algorithm using pyLucene, and save pmids of 50 most similar articles.

P5.candidates_d2v.py (old name: corpus_medline_d2v.py)
   Learn distributed D2V representations on selected titles and abstracts.
   based on: corpus_medline_gensim_D2V.py, corpus_medline_gensim_W2V.py, corpus_medline_D2V_for_MEDLINE_95_97.py

P6.candidates_tfidf.py (old name: corpus_medline_tfidf.py)
   Compute the inversed document frequencies of terms in the vocabulary based on MEDLINE 1985-2015.
   based on: corpus_medline_tfidf_3M.py, corpus_medline_tfidf_for_MEDLINE_95_97.py

P7.candidates_d2v_tfidf_knn.py (old name: corpus_medline_d2v_tfidf_knn.py)
   Find KNN using D2V+TFIDF.
   based on corpus_medline_knn.py, corpus_medline_knn_d2v_tfidf.py, corpus_medline_knn_d2v_tfidf_arc.py, corpus_medline_knn_for_MEDLINE95_97.py

P8.candidates_extract_from_query
   Extract MeSH candidates of L1000, NLM2007, and SMALL200 query PMIDs from their titles and abstracts. No evaluation. Can be generalized to other query data.


A1.candidates_analysis_BM25KNN_from_latest_3M.py (old name: corpus_medline_knn_mesh.py)
   Evaluate the average precision of MeSH candidates generated using BM25 KNN from the latest 3M MEDLINE. Experiments were done on L1000, NLM2007, SMALL200.
   BM25 KNN PMIDs generated from corpus_medline_retrieve.py
   
A2.candidates_analysis_AP_all_reprs.py (old name: corpus_medline_eval.py)
   Evaluate average precisions of MeSH candidates generated from BM25 KNN, D2V KNN, TFIDF KNN, D2V+TFIDF KNN, and BM25+D2V+TFIDF KNN

A3.candidates_analysis_AP_recent_queries.py (old name: corpus_medline_knn_mesh_coverage.py, class Recent_pmids_as_query)
   Analyze the average precision of BM25 KNN candidates of recent PMIDs.

A4.candidates_analysis_AP_knn_general.py (old name: corpus_medline_knn_mesh_coverage.py, class Earlier_pmids_as_ref)
   Analyze the average precision of BM25 KNN candidates from various corpora of different time

A5.candidates_analysis_AP_query_text.py (old name: corpus_medline_mesh_in_query_coverage.py)
   Analyze the average precision of candidates from query texts, using either string match or MetaMap

A6.candidates_analysis_AP_joint_knn_and_query_text.py (old name: corpus_medline_mesh_in_query_coverage.py, corpus_medline_knn_mesh_coverage.py)
   Analyze the average precision of candidates from query texts and BM25 KNN

A7.candidates_analysis_AP_top_terms.py (old name: corpus_medline_top_cand_coverage.py)
   Analyze the average precision of top candidates from joint sets

A8.candidates_analysis_AP_mesh_from_query.py
   Analyze the average precision of candidates extracted from queries. Run after candidates_extract_from_query.py.

A9.candidates_analysis_AP_BM25_knn.py
   Analyze the average precision of BM25 KNN candidates from various corpora of different time.

