# -*- coding:utf-8 -*-
""" Recommend TopN solutions using a content based approach.
"""

from __future__ import print_function
import operator
from scipy.spatial.distance import cosine
from pandas import read_csv
import numpy as np
from my_utils import *
from constants import *
from train import w2vLoad, ldaLoad

w2v_model = w2vLoad() 
if COMBINE_J==False:
    lda_model = ldaLoad()

df = read_csv(DataDir+SlnFile, delimiter='\t', dtype=str)

def recEngine(problem, model_name='wmd'):
    """
    Function
        content based recommendation system. 
    Input Args 
        problem:
            type: str
            desc: user problem
        model_name:
            type: str, default 'wmd'
            desc: differenct approaches to convert sentence to vector. 
                'wmd': word mover distance
                'w2v_max': max(w2v[e] for e in sentence) 
                'w2v_avg': avg(sum(w2v[e] for e in sentence))
                'lda': latent topics of a sentence
    Return
        type: pandas.DataFrame
        desc: TopN solution names, descriptions etc.
    """
    assert(isinstance(problem, str)), ('Invalid input arg: problem')

    problem = string_filter(string_clean(problem))
    problem = tuple(e.strip() for e in re.split(r'[. ]', problem) if e.strip() !='')
    score = 'score_'+str(model_name)
    df[score]=float('inf')

    if model_name=='wmd':
        ### different versions to calculate wmdistance

        # # serial version
        # for idx, s in df.iterrows():
        #     desc = tuple(e.strip() for e in re.split(r'[. ]', s.desc) if e.strip() !='')
        #     df.ix[idx, score] = 1.0/(0.01+w2v_model.wmdistance(problem, desc) )

        # serial version- use "map" (but the speed is almost the same as the serial version)
        idxs = df.index.values
        descs = [tuple(e.strip() for e in re.split(r'[. ]', desc) if e.strip() !='') 
                             for desc in df.desc.values]
        wmds =  map(w2v_model.wmdistance, [problem]*len(idxs), descs)               
        for idx,wmd in zip(idxs,wmds):
            df.ix[idx, score] = 1.0/(wmd+0.01)

        # # parallel version- use "multiprocessing.Pool" (even slower than serial versions)
        # idxs = df.index.values
        # descs = [tuple(e.strip() for e in re.split(r'[. ]', desc) if e.strip() !='') 
        #                      for desc in df.desc.values]
        # from multiprocessing import Pool
        # from functools import partial
        # cal_wmd1 = partial(cal_wmd, problem=problem)
        # p = Pool(processes=2)
        # wmds = p.map(cal_wmd1, descs)
        # for idx,wmd in zip(idxs,wmds):
        #     df.ix[idx, score] = 1.0/(wmd +0.01)

        # # parallel version- use "pp" 
        # # note that "pp" only allow global definition
        # idxs = df.index.values
        # descs = [tuple(e.strip() for e in re.split(r'[. ]', desc) if e.strip() !='') 
        #                      for desc in df.desc.values]
        # import pp
        # job_server = pp.Server(ppservers=())
        # jobs = [job_server.submit(cal_wmd_pp, (desc, problem,w2v_model, ), 
        #         (),())
        #         for desc in descs]
        # for idx, job in zip(idxs, jobs):
        #     df.ix[idx, score] = 1.0/(job()+0.01)

    elif model_name in ['w2v_avg', 'w2v_max', 'lda']:
        idxs = df.index.values
        if model_name == 'lda':
            if COMBINE_J:
                raise Exception("Set constants.COMBINE_J to False to use gensim lda model")
            _, id2word = genCorpus(corpus_name=ModelDir+CORP_NAME)

        def cal_sent2v(sent, model_name):
            """Convert sentence into vector. Needed if using w2v_avg, w2v_max, lda
            """
            if model_name not in ['w2v_avg', 'w2v_max', 'lda']:
                raise ValueError("Invalid input arg: model_name")
            if model_name == 'lda':
                s2v = [0]*TOPICS_NUM
                doc_bow = id2word.doc2bow(sent)
                if doc_bow != []:
                    s2v = lda_model[doc_bow]
                    s2v = [e[1] for e in sorted(s2v, key=operator.itemgetter(0))]
            else:
                w2v_method = {'w2v_avg':np.sum, 'w2v_max': np.max}
                s2v = [0]*VEC_SIZE
                i=0
                for e in sent:
                    try:
                        s2v = w2v_method[model_name]([s2v,w2v_model[e]],axis=0)
                        i+=1
                    except:
                        pass 
                if model_name == 'w2v_avg':
                    s2v = s2v/float(i) if i else s2v
            return s2v  

        p2v = cal_sent2v(problem, model_name)

        # pre-calculate sentence vecotr Sent2Vs using 1. [average|max] of w2v 2. lda
        if 'Sent2Vs' not in globals():
            global Sent2Vs
            Sent2Vs = {'w2v_avg':[], 'w2v_max':[], 'lda':[]} 
        if Sent2Vs[model_name]==[]:        
            for idx, s in df.iterrows():
                desc = (e.strip() for e in re.split(r'[. ]', s.desc) if e.strip() !='')
                desc2v = cal_sent2v(desc, model_name)
                Sent2Vs[model_name].append(desc2v)
        if any(p2v): 
            for idx, s2v in zip(idxs, Sent2Vs[model_name]):
                if any(s2v):
                    if model_name == 'lda':
                        df.ix[idx, score] = 1-hellinger_dist(p2v, s2v)
                    else:
                        df.ix[idx, score] = 1-cosine(p2v, s2v)
    else:
        raise ValueError("Invalid input arg: model_name")

    df[score].replace(float('inf'), float('nan'),inplace=True)
    df1 = df.sort_values(score, ascending=False, inplace=False).head(TOP_N) 
    return df1

def cal_wmd(d, problem):
    return w2v_model.wmdistance(problem,d)

def cal_wmd_pp(d, problem, w2v_model): # pp needs local variable w2v_model
    return w2v_model.wmdistance(problem,d)

def doc2topics(problem, topic_num=float('inf'), words_num = 10):
    """
    Function
        show words under different topics of a document. i.e. doc->topics->words, topic is represented with words
    Input Args
        words_num: maximum number of words to show in each topic
        topic_num: maximum number of topics to show
    """
    problem = string_filter(string_clean(problem))
    problem = tuple(e.strip() for e in re.split(r'[. ]', problem) if e.strip() !='')
    _, id2word = genCorpus(corpus_name=ModelDir+CORP_NAME)
    doc_bow = id2word.doc2bow(problem)
    topic_num = min(topic_num, TOPICS_NUM)
    if doc_bow != []:
        doc_topics = sorted(lda_model[doc_bow], key=operator.itemgetter(1), reverse=True)[:topic_num]
        topic_words = [('%.5f'%topic_prob, [(w1, prob1) for w1,prob1 in lda_model.show_topic(topic_id, topn=words_num)if w1!='']) 
                        for topic_id,topic_prob in doc_topics]
    else: 
        topic_words = [ ('%.5f'%(1.0/TOPICS_NUM),words_probs) for topic_id,words_probs in 
                       lda_model.show_topics(topic_num, formatted=False)[:words_num] ]
    return topic_words




if __name__=='__main__':
    import logging
    import time
    time1 = time.time()
    logging.basicConfig(level=logging.INFO)
    for rs in recEngine(problem='My client wants to solve supply problems'):
        print(rs)
    for rs in recEngine(problem='health care', model_name='w2v_avg'):
        print(rs)
    print('running time: %f'%(time.time()-time1))