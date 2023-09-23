This directories contains the code required for performing the first stage of alignment. The files are meant to be run in the following order -  

## get_similarity  
This computes the similarity scores between the RDF triples and the sentences for a given entity. This can be used for computing both the syntactic similarity and the semantic similarity 
```
python get_similarity.py --candidates_to_path path/to/candidates --langs en --save_dir path/to/save --sim_type <semantic/syntactic>
```
This saves the candidates with the computed similarity scores in a new file 

## get_candidates 
This generates the candidates using a combination semantica and syntactic similarity scores with a KNN classifier  
```
python get_candidates.py --langs en --contrastive_path path/to/contrastive_data --semantic_dict_path path/to/semantic_scores --syntactic_dict_path path/to/syntactic_scores --save_path path/to/save  
```
This generates two files - ```sent_prop_src.txt``` which contains the entity name, sentence number, and language of each generated candidate for bookkeeping purposes, and ```sent_prop_pairs.txt``` which contains the RDF triples and sentence pairs. 
Note that the contrastive data can be generated using the ```generate_contrastive.py``` script in ```utilities``` 

# Utilities  
This contains scripts for finetuning a model on the semantic textual similarity task using contrastive learning which can be useful for computing the semantic similarity between RDF triples and sentences.   
## generative_contrastive.py 
This generates synthetic data which can be used as the training data for contrastive learning  
```
generate_contrastive.py --webnlg_data path/to/webnlg_data --save_name path/to/save 
```
This requires the WebNLG data in the form of separate source and target files.  

## train_contrastive 
This trains a SentenceTransformer model on the STS task using the previously generated data.  

## semantic_score_trusted and syntactic_score_trusted 
These can be used to compute the semantic and syntactic similarity scores for the contrastive data, and can be used to analyze its performance  



