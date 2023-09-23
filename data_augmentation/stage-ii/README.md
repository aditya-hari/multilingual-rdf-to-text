This contains the code required for performing the 2nd stage of alignment using XNLI  

## run_xnli.py 
This performs inference using a pretrained XNLI model on the sentence-triple pairs generated using the previous alignment stage. 

``` 
python run_xnli.py --pairs_path path/to/pairs --save_path path/to/save 
``` 
This generates ```entailments.tsv``` which contains the entailments according to the XNLI model  

## merge_entailments.py 
This merges all the triples which are entailed by a sentence to create the final synthetic corpus 

```
pyhton merge_entailments.py --sent_prop_src_path path/to/sent_prop_src.txt --entailments_path path/to/entailments.txt --save_path path/to/save
``` 

This requires the sent_prop_src.txt file generated in stage 1  
