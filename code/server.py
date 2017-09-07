# -*- coding:utf-8 -*-
"""Build API, start http server
"""

from constants import *
from flask import Flask, request, jsonify
import logging
from rec_engine import recEngine
import sys

app = Flask(__name__) # create a Flask class

# create API 
@app.route('/problem', methods=['GET'])
def recommend_sln():
    logging.info('Get request')
    user_problem = request.args[INPUT_ARG]
    if sys.version_info[0]==2:
        user_problem = user_problem.encode('utf8')
    logging.info('Do recommendation')

    # show 'w2v_avg' result
    s2v_method = 'w2v_avg'
    rec_rs = recEngine(user_problem, s2v_method)
    rs = {}
    rs.setdefault('t1', []) 
    rs['t1'].append('\t'.join(['id','Name','Score','Description']))
    for _,j in rec_rs.iterrows():
        # rs.setdefault(j['id'], str(j.id_name)+'\t'+str(j.desc))
        rs['t1'].append('\t'.join([j.id, j.id_name, str(j.score), 
                                '. '.join(j.desc.split('.')[:3]) ]))

    # show 'lda' result
    s2v_method = 'lda'
    rec_rs = recEngine(user_problem, s2v_method)
    rs.setdefault('t2', [])
    rs['t2'].append('\t'.join(['id','Name','Score','Description']))
    for _,j in rec_rs.iterrows():
        # rs.setdefault(j['id'], str(j.id_name)+'\t'+str(j.desc))
        rs['t2'].append('\t'.join([j.id, j.id_name, str(j.score), 
                                '. '.join(j.desc.split('.')[:3]) ]))

    logging.info('Return results')
    return jsonify(rs)

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    app.run(host='0.0.0.0', port=PORT, debug=True, threaded=True)
