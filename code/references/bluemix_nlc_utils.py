# -*- coding: utf-8 -*-
"""
Created on Thu Apr 13 09:31:01 2017
@author: jasonm
"""

# import json
import logging
# from os.path import join, dirname
from watson_developer_cloud import NaturalLanguageClassifierV1


class Bluemix_Util():
    # Global logger set here
    _mlogger = None

    def __init__(self):
        """ Create the util object. Set Object vars to point
        to the Bluemix Based NLC's
        """

        # Credentials for CAO Dedicated Bluemix NLC trained classifiers
        # Ref:
        self.nlc = NaturalLanguageClassifierV1(
            username='e3c6cf5a-90d9-4dd3-acc2-322f3bd8f660',
            password='BFQgPQ0iZLsD')

        # Get the classifier ID's form this
        # Here we identify manually to get up and running
        # ToDo: Automatically find the id from a classifier "name"
        self.classifiers = self.nlc.list()
        # print(json.dumps(classifiers, indent=2))

        # ToDo: Pull out classifier id from above json
        self.nlc_role = 'f5bbbbx174-nlc-5814'
        self.nlc_industry = 'f5bbbbx174-nlc-5812'
        self.nlc_outcome = 'f5bbbcx175-nlc-5826'
        self.nlc_symptom = '4d5c10x177-nlc-1599'
        self.nlc_solution = '90e7b7x198-nlc-332'

    def set_logger(self, logger_to_use):
        """ Depcrecated: Use single logger
        Instantiate our local logger here.
        Import this module, setup logger from parent, call to use here
        """
        # global mlogger
        self._mlogger = logger_to_use
        print("setting the logger inside bm_nlc")
        self._mlogger.info("Setting the logger")  # inside %s" % (__name__))

    # create a classifier
    # with open('../resources/weather_data_train.csv', 'rb') as training_data:
    #     print(json.dumps(nlc.create(
    # training_data=training_data, name='weather'), indent=2))

    def get_role(self, text):
        return(self._get_nlc_classification(text, self.nlc_role))

    def get_industry(self, text):
        return(self._get_nlc_classification(text, self.nlc_industry))

    def get_outcome(self, text):
        return(self._get_nlc_classification(text, self.nlc_outcome))

    def get_symptom(self, text):
        return(self._get_nlc_classification(text, self.nlc_symptom))

    def get_solution(self, text):
        return(self._get_nlc_classification(text, self.nlc_solution))

    def _get_nlc_classification(self, text, nlc_id):
        """ Send the text to the given classifier id and return
        top {class, confidence} only
        """
        status = self.nlc.status(nlc_id)
        # mlogger.info(json.dumps(status, indent=2))

        if status['status'] == 'Available':
            # response contains dictionary
            # Pull out the identified classes.
            classes = self.nlc.classify(nlc_id, text)
            # top_prediction = classes['classes'][0]
            # print(json.dumps(classes, indent=2))
            # self._mlogger.info("nlc_id %s" % (nlc_id))
            logging.info("nlc_id %s" % (nlc_id))
            logging.info("Classes = %s" % (classes))
            return(classes)
        else:
            # self._mlogger.info("NLC with id %s is not available" % (self.nlc_id))
            logging.info("NLC with id %s is not available" % (self.nlc_id))
            return()

# delete = nlc.remove('2374f9x68-nlc-2697')
# print(json.dumps(delete, indent=2))

# example of raising a WatsonException
# print(json.dumps(
#     nlc.create(training_data='', name='weather3'),
#     indent=2))


def self_test():
    """ Self test here
    """
    logging.basicConfig(level=logging.INFO)
    logging.info('Started')
    bm = Bluemix_Util()
    # bm.set_logger(mlog)
    bm.get_role("This telco company ceo interested in finance regulation")
    logging.info('Finished')


if __name__ == '__main__':
    """ If run directly just do self test
    """
    self_test()
