# -*- coding:utf-8 -*-
"""Useful tools
"""

from __future__ import print_function
import re
import sys
if sys.version_info[0]==2:
    import cPickle as pickle
else:
    import pickle
from os import path
from math import sqrt
from nltk import PorterStemmer
from nltk.corpus import stopwords
from gensim.corpora import dictionary

StopWords = stopwords.words('english')

def string_clean(s, level='weak', keep_dot = False):
    """ 
    Function
        1. eliminate abnormal characters, determined by arg 'level'
        2. substitute "? !" with "."; substitute ",/:" with ' '
        3. lower() all words
    Input Args: 
        s:
            type: str
            desc: raw text, usually a long string 
        level: 
            type: str, default 'weak'
            desc: level to eliminate abnormal characters
                  'weak': url tags like "<...>" or "&...;", illegal characters and blank strings
                  'strong': all characters except a-zA-Z
        keept_dot:
            type: str, default 'false'
            desc: 
    Return:
        type: str
        format: a long string containing many sentences, ' ' between words (and if keep_dot is Ture: with '.' between sentences) 
    Usage: string_clean('&amp This IS a </test> File! &amp;')
    """
    s = re.sub(r'[\?!:,]','.',s)
    if level=='strong':
        s_pattern = r'[^.a-zA-Z]'
    else:
        s_pattern = r'<.*?>|&.*?;|\-+|\_+|[/%<>=~\+\|\*\\[\]\(\)\n\r\t\\]'
    s = re.sub(s_pattern, ' ',s)
    res = []
    for sent in s.strip().split('.'):
        if sent.strip() != '':
            sent_l = []
            for w in sent.strip().split(' '):
                if w != '':
                    try:
                        if sys.version_info[0] == 2: # for python2
                            sent_l.append(w.strip().lower().encode('utf-8'))
                        elif sys.version_info[0] == 3:
                            sent_l.append(w.strip().lower()) # for python3
                        else:
                            raise RuntimeError('Python version error')
                    # ignore illegal characters
                    except:
                        pass
            res.append(' '.join(e for e in sent_l if e!=''))
    if keep_dot:
        return '.'.join(res)
    else:
        return ' '.join(res)


def string_filter(sents, level = 'do_stemming_stopwords'):
    """
    Function
        filter stop words and stemming 
    Input Args
        sents 
            type: str 
            desc: can be output of string_clean(), format like 'hansome guy.do better in your area.go go go'
        level
            type: str, default 'do_stemming_stopwords'
            desc: different levels to process cleaned text
                  'do_stemming': stemming cleaned text
                  'do_stopwords': filter stopwords from cleaned text
                  'do_stemming_stopwords': do both
    Return
        type: str
        format: a long string containing many sentences, with ('.' between sentences and) ' ' between words
    """
    if level=='do_stemming_stopwords':
        stemmer = PorterStemmer()
        return '.'.join(' '.join(stemmer.stem(w.strip()) for w in sent.strip().split(' ') if w.strip() not in StopWords ) 
                        for sent in sents.strip().split('.') )
    elif level=='do_stopwords':
        return '.'.join(' '.join(w.strip() for w in sent.strip().split(' ') if w.strip() not in StopWords ) 
                        for sent in sents.strip().split('.') )
    elif level=='do_stemming':
        return '.'.join(' '.join(stemmer.stem(w.strip()) for w in sent.strip().split(' ') ) 
                        for sent in sents.strip().split('.') )
    else:
        print('Warning: Do nothing to string')
        return sents

class SentenceIter(object):
    """
    Function
        iteratively read files in line
    Usage
        docs = SentenceIter(file1, file2, ...)
    """
    def __init__(self, *fnames):
        self.fnames = fnames
 
    def __iter__(self):
        for fname in self.fnames: 
            for line in open(fname,'r'):
                line = string_clean(line, level='strong')
                yield line.strip().split(' ')

def genCorpus(corpus_name, f_inputs=[], word_count = 20, word_doc_freq=0.3):
    """
    Function
        generate corpus (bow representation of documents, id2words) for topic models(e.g. LDA)
    Input Args
        f_inputs
            type: list of filenames
            format: ['file1', 'file2', ...]
            desc: files to be converted to corpus
        corpus_name
            type: str
            desc: corpus path to be stored
    Return
        gensim_docBow
            type: list
            format: [ [doc1 bag of words],
                      [doc2 bag of words],
                     ]
        id2word
            type: gensim.corpora.Dictionary
    """
    if not path.exists(corpus_name):
        docs = SentenceIter(*f_inputs)
        wordDict = dictionary.Dictionary(documents=docs, prune_at=None)
        wordDict.filter_extremes(no_below=word_count, no_above=word_doc_freq) # filter word count< WORD_COUNT, word appear in documents rate > WORD_DOC_FREQ
        corpus_docBow = [wordDict.doc2bow(doc) for doc in docs]
        with open(corpus_name, 'wb') as fo:
            pickle.dump([corpus_docBow, wordDict], fo)
    else:
        with open(corpus_name, 'rb') as fi:
            corpus_docBow, wordDict =  pickle.load(fi)   
    return corpus_docBow, wordDict

def hellinger_dist(v1, v2):
    """
    Function
        calculate hellinger distance, which is more useful for similarity between 
        probability distributions (such as LDA topics)
        see https://en.wikipedia.org/wiki/Hellinger_distance for hellinger_dist
    Input Args
        v1|v2
            type: list, with the same size. each element in range [0,1]
            desc: represent a vector
    Return
        type: float, range: [0,1]
    """
    if len(v1) != len(v2):
        raise ValueError("Vectors should have the same size! ")
    return sqrt( sum( map(lambda e: 
                        (sqrt(e[0])-sqrt(e[1]))**2, zip(v1,v2))))/sqrt(2)


if __name__=='__main__':
    print(string_filter(string_clean("everyday...I enjoy. ? 14 seating here", level='strong', keep_dot=True)))
