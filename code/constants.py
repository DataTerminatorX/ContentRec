# -*- coding:utf-8 -*-
"""Store Constants
"""


### Global Constants
ModelDir = '../model/'
DataDir = '../data/'
CPU_NUM = 1
CORP_NAME = 'docsBow' # file name to save generated gensim corpus

### Training Files
TrainFiles =[] # add your training file here

### Source File
SourceFile = '' # add your source file here

### Model Parameters
MIN_COUNT = 5
VEC_SIZE = 2000
CONTEXT_WINDOW = 15
W2V_TYPE = 0
W2V_ModelName = 'w2v_size%s_window%s_minCount%s'%(VEC_SIZE, CONTEXT_WINDOW, MIN_COUNT)

TOPICS_NUM = 20
ITERS = 200
CONVERG_TH = 0.000001
if CPU_NUM > 1:
    ALPHA = 0.1
else:
    ALPHA = 'auto'
ETA = 'auto'
MIN_PROB = 0
CHUNK_SIZE = 1000
PASSES=5
LDA_ModelName = 'lda_topicSize%s_alpha%s_eta%s_iter%s_convergTh%s_passes%s'%(
                TOPICS_NUM, ALPHA, ETA, ITERS, CONVERG_TH, PASSES)


TOP_N = 10

### Server Parameters
PORT = 5000
INPUT_ARG = 'text' 