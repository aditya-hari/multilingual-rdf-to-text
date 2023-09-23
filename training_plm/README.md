The contains the codes for denoising the synthetic data and training pretrained language models iteratively on the denoised data 

## scoring 
This generates the logprob scores for a corpus of triple-sentence pairs using a language model. 

```
python --lang en --dataset_path path/to/data --model_checkpoint path/to/model --save_path path/to/save 
``` 
The data is expected in the form of separate source and target files in the dataset directory. This can be used to compute the scores using the noisy model and the denoised model  

## train_baseline 

This trains a baseline model using pretrained models from Hugginface. This can be used to train a noisy model by training a model on the synthetic dataset, and can be further used to denoise that model by training it on the trusted data 

## train_annealing 
This trains a model in an iterative manner by progressively exposing to higher quality data. This requires the logprob scores of every sample scored by a noisy model and a denoised model. 
