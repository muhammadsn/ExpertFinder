#!/usr/bin/python

import sys, getopt, os
from ExpertFinder import Indexer
from ExpertFinder import Retriever
from ExpertFinder import Evaluator


def main(argv):

    query = ""
    _lambda = -1
    dir_sep = "/"
    settings = {
        'project_path': os.path.dirname(os.path.realpath(__file__)) + dir_sep,
        'resource_path': os.path.dirname(os.path.realpath(__file__)) + dir_sep + "Resources" + dir_sep,
        'index_path': os.path.dirname(os.path.realpath(__file__)) + dir_sep + "Index" + dir_sep,
        'result_path': os.path.dirname(os.path.realpath(__file__)) + dir_sep + "Results" + dir_sep,
        'eval_path': os.path.dirname(os.path.realpath(__file__)) + dir_sep + "Evals" + dir_sep,
        'doc_ranking_path': os.path.dirname(os.path.realpath(__file__)) + dir_sep + "Resources" + dir_sep + "doc_rankings" + dir_sep,
        'data_file_name': "Posts.xml",
        'dataset_file_name': "dataset.json",
        'index_main_file_name': "all-postings.json",
        'index_file_name': "postings",
        'judgement_file_name': "GoldenSet.csv",
        'stemmer': "porter",
        'eval_metrics': ['MAP', 'p10', 'MRR'],
        'request_count': 1000,
        'indexing': False,
        'retrieval': False,
        'evaluation': False,
    }

    try:
        opts, args = getopt.getopt(argv, "hireq:l:", ["query=", "lambda="])
    except getopt.GetoptError:
        print('[!] USAGE: python3 main.py -[i|r|v] -q "<query-string>" -l <lambda-value>')
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print('[!] USAGE: python3 main.py -[i|r|v] -q "<query-string>" -l <lambda-value>')
            sys.exit()
        elif opt == '-i':
            settings['indexing'] = True
        elif opt == '-r':
            settings['retrieval'] = True
        elif opt == '-e':
            settings['evaluation'] = True
        elif opt in ("-q", "--query"):
            query = arg
        elif opt in ("-l", "--lambda"):
            _lambda = float(arg)

    if query == "" or _lambda < 0:
        print('[!] One or more arguments (query or lambda) missing...')
        print('[!] USAGE: python3 main.py -[i|r|v] -q "<query-string>" -l <lambda-value>')
        sys.exit(2)
    elif not (settings['indexing'] or settings['retrieval'] or settings['evaluation']):
        print('[!] No Action Specified, Nothing to Do!')

    if settings['indexing']:
        A = Indexer(resource_path=settings['resource_path'], data_file_name=settings['data_file_name'],
                    index_path=settings['index_path'], index_file_name=settings['index_file_name'],
                    dataset_file_name=settings['dataset_file_name'], stemmer=settings['stemmer'])

    if settings['retrieval']:
        B = Retriever(resource_path=settings['resource_path'], result_path=settings['result_path'],
                      doc_ranking_path=settings['doc_ranking_path'], dataset_file_name=settings['dataset_file_name'],
                      index_path=settings['index_path'], index_file_name=settings['index_main_file_name'],
                      stemmer=settings['stemmer'], query=query, req_count=settings['request_count'], lmbd=_lambda)

    if settings['evaluation']:
        C = Evaluator(resource_path=settings['resource_path'], result_path=settings['result_path'],
                      evaluation_path=settings['eval_path'], judgement_file_name=settings['judgement_file_name'],
                      query=query, metrics=settings['eval_metrics'])


if __name__ == '__main__':
    main(sys.argv[1:])
