#!flask/bin/python
"""
sample code from Jason Mallen
"""

from flask import Flask, render_template, request, jsonify
import os
import bluemix_nlc_utils as bm_util
import requests
import logging
# from logging.handlers import RotatingFileHandler
# import json

app = Flask(__name__)
bm = bm_util.Bluemix_Util()

if os.getenv("VCAP_APP_PORT"):
    port = int(os.getenv("VCAP_APP_PORT"))
else:
    port = 8080


# start an html, can input text in html, send to '/advisor/api/v0.1/sendtext' and get response
# do all those things within python, needn't use javascript
@app.route('/', methods=['GET'])
def display_template():
    if "lower" in request.args:
        return request.args["lower"].lower()
    elif "upper" in request.args:
        return request.args["upper"].upper()
    else:
        return render_template('index.html')

# an api method
@app.route('/advisor/api/v0.1/sendtext', methods=['GET'])
def send_text():
    """  Send the user text to our identifiers """
    logging.info("Send text to Bluemix Services")
    # the 'text' keyword is defined in front end, where you can input text and call api like
    # http://local host:8080/advisor/api/v0.1/sendtext?text='I like eating food'
    user_text = request.args["text"]  

    role_d = bm.get_role(user_text)
    # logging.info("Role response = %s" % (role_d))
    industry_d = bm.get_industry(user_text)
    outcome_d = bm.get_outcome(user_text)
    solution_d = bm.get_solution(user_text)
    symptom_d = bm.get_symptom(user_text)

    # build our response content
    # respond with dictionary of dictionaries
    # each contains the entity:probability

    evidence_d = {'role': role_d,
                  'industry': industry_d,
                  'outcome': outcome_d,
                  'solution': solution_d,
                  'symptom': symptom_d}

    logging.info("evidence = %s" % (evidence_d))
    # send the evidence to the inference engine - the response is list of
    # solns, each soln a dictionary
    soln_rec = send_evidence(evidence_d)
    logging.info("Solution recommendation is %s" % (soln_rec))
    # create the return package json object containing
    # 2 objects { evidence, solns}
    resp_package = {'evidence': evidence_d, 'soln': soln_rec}

    logging.info("Response Package containing: %s" % (resp_package))
    # response with json
    return(jsonify(resp_package))


def send_to_r_engine(evidence_d):
    """
    Send the evidence object to the R inference engine web service.

    :param object evidence_d:

    """
    _url = "https://inference-engine-cao.w3ibm.mybluemix.net/custom/postEvidence"
    # change http to https if the above link doesn't work

    # sample payload here - use for testing. Provided by Dong Sheng in docs.
    payload = {"type": "classifiers",
               "industry": [{"shortName": "AerospacDefense", "probability": 0.22674095619789802},
                            {"shortName": "MediaEntert", "probability": 0.14632511392462516},
                            {"shortName": "Telco", "probability": 0.120634875636208},
                            {"shortName": "TravelTransp", "probability": 0.09051849254616975},
                            {"shortName": "Insurance", "probability": 0.06792063613129566},
                            {"shortName": "Healthcare", "probability": 0.04353191599669933},
                            {"shortName": "Retail", "probability": 0.041811164199502066},
                            {"shortName": "ChemPetrol", "probability": 0.03599039794774384},
                            {"shortName": "Electronics", "probability": 0.03206674163725133},
                            {"shortName": "InduProds", "probability": 0.02857084049815816}],
               "role": [{"shortName": "CEO", "probability": 0.928695509047156},
                        {"shortName": "Strategy", "probability": 0.014914324657187263},
                        {"shortName": "Security", "probability": 0.007316772105995695},
                        {"shortName": "Finance", "probability": 0.007025085804438148},
                        {"shortName": "Marketing", "probability": 0.005652827841901833},
                        {"shortName": "Sales", "probability": 0.00509915667016486},
                        {"shortName": "CAO", "probability": 0.004619907620707702},
                        {"shortName": "Technology", "probability": 0.0034373627065985954},
                        {"shortName": "HR", "probability": 0.003187346838156185},
                        {"shortName": "Data", "probability": 0.0031367183669213034}],
               "symptom": [{"shortName": "S_9_97_0", "probability": 0.42863662033994315},
                           {"shortName": "S_11_131_13", "probability": 0.04313536404343483},
                           {"shortName": "S_11_131_14", "probability": 0.03184416786505809},
                           {"shortName": "S_5_54_7", "probability": 0.028525957430403703},
                           {"shortName": "S_3_29_3", "probability": 0.025711095255416917},
                           {"shortName": "S_11_131_9", "probability": 0.01999763964066662},
                           {"shortName": "S_9_99_4", "probability": 0.01657815581516856},
                           {"shortName": "S_3_36_2", "probability": 0.016417582246304737},
                           {"shortName": "S_4_43_0", "probability": 0.016258563969314452},
                           {"shortName": "S_16_175_9", "probability": 0.013746209232668536}],
               "outcome": [{"shortName": "O_11_131", "probability": 0.10449796393405533},
                           {"shortName": "O_10_108", "probability": 0.08461967174406515},
                           {"shortName": "O_9_97", "probability": 0.06852275945387847},
                           {"shortName": "O_15_253", "probability": 0.05548790802894364},
                           {"shortName": "O_13_252", "probability": 0.05139301171963129},
                           {"shortName": "O_12_238", "probability": 0.04226802920839994},
                           {"shortName": "O_9_245", "probability": 0.03928008944088379},
                           {"shortName": "O_17_227", "probability": 0.03399608369541746},
                           {"shortName": "O_13_158", "probability": 0.02945650721950785},
                           {"shortName": "O_key", "probability": 0.028977705221669578}],
               "solution": [{"shortName": "BS_5_53", "probability": 0.1441746599398951},
                            {"shortName": "BS_10_249", "probability": 0.04080998330444033},
                            {"shortName": "BS_10_107", "probability": 0.040809890099611085},
                            {"shortName": "BS_16_175", "probability": 0.040304380932563943},
                            {"shortName": "BS_9_245", "probability": 0.03558310109118805},
                            {"shortName": "BS_16_181", "probability": 0.03520056716816222},
                            {"shortName": "BS_9_99", "probability": 0.024604941401158542},
                            {"shortName": "BS_12_141", "probability": 0.024494256099436687},
                            {"shortName": "BS_9_97", "probability": 0.024384068715422397},
                            {"shortName": "BS_3_30", "probability": 0.024132727281989345}]
               }

    r = requests.post(_url, json=payload)
    return(r.json())


def send_evidence(evidence_d):
    """ Placeholder/dummy fn.
    here we pretend to find solutions and create a dummy data struct
    in response.
    """

    sol_1 = {'soln_id': 'uid3456',
             'short-desc': 'Enterprise Analytics Fraud',
             'asset_link': 'http://bit.ly/123',
             'confidence': 0.89}

    sol_2 = {'soln_id': 'uid7890',
             'short-desc': 'Health Analytics Patient',
             'asset_link': 'http://bit.ly/001',
             'confidence': 0.32}

    # list of solns (could be empty)
    soln_rec = [sol_1, sol_2]
    return(soln_rec)


if __name__ == '__main__':
    # setup logging - use app.logger.[warning,info,error]('our log msg')
    # handler = RotatingFileHandler('advisor_api.log', maxBytes=100000, backupCount=5)
    # handler.setLevel(logging.INFO)
    # app.logger.addHandler(handler)
    # send the logger to our helper modules
    # bm_nlc.set_logger(app.logger)
    logging.basicConfig(level=logging.INFO, )
    logging.info('Advisor Flask Rest API Starting...')

    app.run(host='0.0.0.0', port=port, debug=True)
