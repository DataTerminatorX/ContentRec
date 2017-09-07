# -*- coding: utf-8 -*-
"""Parse and process text extracted from .txt/.pdf/.pptx/.doc/.docx files
"""

import textract
import os
from my_utils import string_clean, string_filter
import logging


def get_all_filenames(dir_name):
    """traverse all files in a nested dir_name
    """
    for root_dir, sub_dirs, files in os.walk(dir_name):
        for filename in files:
            yield root_dir, filename

def parse_files(dir_name, target_file_name, clean_level = 'weak', filter_level=None, folder_level='sub_folder'):
    """
    Function
        Parse and process text from each file in a nested folder
    Input Args:
        dir_name
            desc: root folder directory, store files/folders to be parsed
        target_file_name
            desc: target file path, store text extrated from files
        clean_level
            type: str, default 'weak_clean'
            desc: different levels to process text extracting from files
                  'raw': raw text extracted from files
                  'weak_clean' or other value: filter abnormal characters from raw text, see my_utils.string_clean
                  'strong_clean': filter all characters except a-zA-Z, see my_utils.string_clean
        filter_level
            type: str, default None
            desc: see my_utils.string_filter
                  'do_stopwords': filter stopwords from cleaned text
                  'do_stemming'
                  'do_stemming_stopwords'
        folder_level
            type: str, default 'file'
            desc: will affect output 
                'file': treat each file as a document
                'sub_folder': treat all files in a sub-folder as a document
    Return:
        desc: a .txt file to store results
        format: 
            line_i: string of path, determined by arg 'folder_level'
            line_i+1: a long string, containing text extracted from files. 
                      format determined by arg 'level'
            line_i+2: \n
    """
    d = {}
    for f_path, f in get_all_filenames(dir_name):
        file_format = f.split('.')[-1]
        f_full = f_path + '/' + f
        if file_format == 'ppt':
            # currently fail to parse .ppt files
            logging.info('Warning: ignore file: %s, unsupport .ppt file' % f_full)
            continue
        elif  file_format in ['pdf', 'doc', 'docx', 'pptx']: 
            text = textract.process(f_full, method='pdfminer')
        else:
            # files in other format (.txt etc.) or without format
            with open(f_full, 'r') as fi:
                text = fi.read()
        logging.info('Processing file: %s' % f_full)
        if clean_level=='raw':
            pass
        else:
            text = string_clean(text, level = clean_level)
        text = string_filter(text, level=filter_level)
        if folder_level=='file':
            d.setdefault(f_full, '')
            d[f_full] += text
        else:
            d.setdefault(f_path, '')
            d[f_path] += text            

    with open(target_file_name, 'w') as fo:
        for k, v in d.items():
            print >> fo, k
            print >> fo, v + '\n'


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    # parse_files('../data/Conversations', '../data/data_merged.txt')
    parse_files('../data/20news-18828', '../data/20news_merged4dongsheng.txt', 
                clean_level='strong',filter_level='do_stemming_stopwords', folder_level = 'file')
