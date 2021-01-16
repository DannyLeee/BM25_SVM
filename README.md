## Introduction
* This is the Information Retrieval final project.
* Methodology based on *Learning to rank for determining relevant document in Indonesian-English cross language information retrieval using BM25*.

## Usage
```sh
code.py [-h] -mode
               {train_scratch,train,test,preprocess,get_feature,get_svm_data}
               [-doc_dict_path DOC_DICT_PATH] [-hidden_size HIDDEN_SIZE]
               [-bm_scratch BM_SCRATCH] [-feature_word_path FEATURE_WORD_PATH]
               [-df_path DF_PATH] [-svm_data_path SVM_DATA_PATH]
               [-lexicon_path LEXICON_PATH] [-model_mode {train,test}]
               [-model_path MODEL_PATH]

optional arguments:
  -h, --help            show this help message and exit
  -mode {train_scratch,train,test,preprocess,get_feature,get_svm_data}
  -doc_dict_path DOC_DICT_PATH
                        (default: data/document.pkl)
  -hidden_size HIDDEN_SIZE
                        feature words size (default: 50)
  -bm_scratch BM_SCRATCH
                        select feature words from scratch or not (default: 0)
  -feature_word_path FEATURE_WORD_PATH
                        file path to save feature_words after selecting
  -df_path DF_PATH      preprocess/testing dataframe path (.csv)
  -svm_data_path SVM_DATA_PATH
                        file path to save svm_data after preprocess or load when testing
  -lexicon_path LEXICON_PATH
                        feature_words file path
  -model_mode {train,test}
                        decide the data will be processed in which mode
  -model_path MODEL_PATH
                        the model path to save or load (.pkl)
```
### mode introduce
* train_scratch: Training model from scratch. Needed argument: `hidden_size`, `df_path`, `model_mode`, `model_path`
* train: Use training data(`svm_data_path`) to train model. Model will save at `model_path`.
* test: Use testing data(`svm_data_path`, `df_path`) to test model(`model_path`)
* preprocess: Get document dictionary from raw data. File will save at `doc_dict_path`.
* get_feature: Get `hidden_size` of feature words from document dictionary. File will save at `feature_word_path`.
* get_svm_data: Get svm input data from document dictionary(`doc_dict_path`), feature words(`lexicon_path`) and query dataframe(`df_path`). File will save at `svm_data_path`.

## Methodology
* Use index terms' BM25 score to select feature words.
* Feature words and query's words as a document-query pair's feature vector.
* The feature vector use a SVM (with kernel `rbf`) to classify relevance.
* Finally use original BM25 score and SVM result to sort the relevance.

## Acknowledge
* Thank [@Striper](https://github.com/justbuyyal) doing data prerpocess; [@YuKai Lee](https://github.com/leeyk0501) and [@Jeffery Ho](https://github.com/chiachun2491) doing feature words selecting; [@Jeffery Ho](https://github.com/chiachun2491) doing  presentate.

<!-- <img src="https://latex.codecogs.com/gif.latex?[formula]"/> -->