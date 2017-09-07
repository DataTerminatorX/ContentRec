# A Content-based Recommender Framwork

This framework aims at recommending items to users based on the similarities of items' descriptions and users' requests. An api is provided to make this framework embedded in back-end.

Rememeber to alter parameters and constants if you are using your own source data files. Usually the file is in below format

|id|name|description|
|:-----|:-------|:-------|
|01|Wine|Made in China|

[toc]

## 1. Features
1. Compatible with python2.7.x and python3.5.x (**python3 is preferred**)
2. Optimized for quick response (Usually within 2 seconds)
3. Parallel model training
4. File parser for external files in `.txt/.pdf/.pptx/.doc/.docx` format.

### Development Process

[] Implementation of various sentence/paragraph encoding models
    [] Unsupervised models
        [x] average of word2vec
        [x] max of word2vec
        [x] lda topics
        [x] word mover distance(paper: "From Word Embeddings To Document Distances")
        [] topic word embedding(paper: "Topical Word Embeddings")
    [] Supervised models
        [] seq2seq model(paper: "The Ubuntu Dialogue Corpus- A Large Dataset for Research in Unstructured Multi-Turn Dialogue Systems")
[x] An api for depolying on server

## 2. Running Procedure
### 2.1. Installing requirements

```python
pip install -r requirements.txt
```
Download stopwords by entering `python` in terminal window and using `nltk.download()`

### 2.2. Setting constants

If needed, change constants (e.g. cpu number, server port, word2vec/lda model parameters) in `constants.py`. BTW, model parameters have already been tuned.

* From my experience, parameters which are vital important to models are: `TrainFiles`, `VEC_SIZE`, `W2V_TYPE`, `TOPICS_NUM`, `PASSES`, `WORD_DOC_FREQ`, 

### 2.3. Starting server
Run `server.py` on server. Will create a website like `http://[your ip address]:5000/` and an API like `http://[your ip address]:5000/problem?text=`

### 2.4. Usage
Input a text string through html's text input box, or through API (e.g. `http://[your ip address]:5000/problem?text=Detect fraud and cheat in ledger`), the API will send recommendation results in json format

## 3. File Structure
* **code** 
(see comments in each code file for details)
    * `server.py`: Build an API and start a http server
    * `templates`: Html and js file.
    * `rec_engine.py`: Do recommendation
    * `train.py`: Train word2vec/lda model. Save models in **model** folder
    * `my_utils.py`: General functions. Including text processing, corpus generation, etc.
    * `constants.py`: Set global parameters
    * `file_parser.py`: Extract text from documents
    * **refrences**
        * `upper.py`: Use flask to build a server API (Jason sent me)
        * `bluemix_nlc_utils.py`: Configure bluemix NLC API in python (Jason sent me)
* **data**: Empty. For storing data files
* **model**: Empty. For stroing models
* **docs**
    * `*.html`: Auto generated docs for code modules.

