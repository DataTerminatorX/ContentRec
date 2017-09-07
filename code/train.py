# -*- coding:utf-8 -*-
"""Train and load word2vec/lda models
"""

from __future__ import print_function
from gensim.models import word2vec, ldamodel, ldamulticore
from gensim.models.wrappers import ldamallet
from os import path
import logging
from constants import *
import numpy as np
from my_utils import SentenceIter, genCorpus

logging.basicConfig(level=logging.INFO)
# fix random seed
np.random.seed(seed=38)

def w2vTrain(f_inputs, model_output, **args): 
    """
    Input Args
        f_input
            training corpus, one sentence a line with ' ' seperated
        model_input
            model name
        **args
            contain 'min_count', 'workers', 'size', 'window'
    """
    sentences = SentenceIter(*f_inputs)
    w2v_model = word2vec.Word2Vec(sentences, **args)
    w2v_model.save(model_output, ignore=['syn1neg'])

def w2vLoad():
    """
    Train and load existing word2vec model
    """
    if not path.exists(ModelDir+W2V_ModelName):
        files = [DataDir+TrainFile for TrainFile in TrainFiles]
        w2vTrain(files, ModelDir+W2V_ModelName,
                min_count=MIN_COUNT, workers=CPU_NUM, size=VEC_SIZE, window=CONTEXT_WINDOW,
                sg=W2V_TYPE) 
    return word2vec.Word2Vec.load(ModelDir+W2V_ModelName)

def ldaTrain(f_inputs, model_output, corpus_name, **args):
    """
    Input Args
        f_input
            training corpus, one document a line with ' ' seperated
        model_input
            model name
        corpus_name
            convert training corpus to gensim format corpus
        **args
            contain 'num_topics', 'alpha', 'eta'
    """
    lda_corpus, id2word = genCorpus(corpus_name=corpus_name, f_inputs=f_inputs)
    if CPU_NUM > 1:
        lda_model = ldamulticore.LdaMulticore(lda_corpus, workers=CPU_NUM, id2word=id2word, **args)
    else:
        lda_model = ldamodel.LdaModel(lda_corpus, id2word=id2word, **args)
    lda_model.save(model_output, ignore=['state', 'dispatcher'])

def ldaLoad():
    """
    Train and load existing lda model
    """
    if not path.exists(ModelDir+LDA_ModelName):
        files = [DataDir+TrainFile for TrainFile in TrainFiles]
        ldaTrain(files, ModelDir+LDA_ModelName, ModelDir+CORP_NAME,
                num_topics=TOPICS_NUM, alpha = ALPHA, eta = ETA, minimum_probability=MIN_PROB, iterations = ITERS, gamma_threshold = CONVERG_TH,
                chunksize=CHUNK_SIZE, passes=PASSES,
                )
    if CPU_NUM > 1:
        return ldamulticore.LdaMulticore.load(ModelDir+LDA_ModelName)
    else:
        return ldamodel.LdaModel.load(ModelDir+LDA_ModelName)

def ldaMalletLoad():
    '''
    Train and load existing lda(mallet) model
    Note: lda(mallet) model has an unfixed problem, see "4.3.2 gensim:..." in README.md
    '''
    if not path.exists(ModelDir+LDA_MalletModelName):
        lda_corpus, id2word = genCorpus(corpus_name=ModelDir+CORP_NAME, f_inputs=[DataDir+TrainFile for TrainFile in TrainFiles])
        lda_model = ldamallet.LdaMallet(ModelDir+'mallet-2.0.8/bin/mallet', corpus=lda_corpus, 
            num_topics=TOPICS_NUM, id2word=id2word, workers=CPU_NUM, iterations=ITERS)
        lda_model.save(ModelDir+LDA_MalletModelName)
    return ldamallet.LdaMallet.load(ModelDir+LDA_MalletModelName)




if __name__=='__main__':
    # w2vLoad()
    ldaLoad()