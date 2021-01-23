import numpy as np
import pandas as pd
from .fileHandler import Importer as load
from .fileHandler import Exporter as save


class Evaluator:

    settings = {
        'resource_path': "",
        'result_path': "",
        'evaluation_path': "",
        'judgement_path': "",
    }

    query = ""
    metrics = []
    k = 10
    judgements = pd.DataFrame()
    results = {}
    results_info = {}
    rel_ca_list = []
    rel_count = 0

    def __init__(self,  resource_path, result_path, evaluation_path, judgement_file_name, query, metrics):
        self.settings['resource_path'] = resource_path
        self.settings['result_path'] = result_path
        self.settings['evaluation_path'] = evaluation_path
        self.settings['judgement_path'] = resource_path + judgement_file_name
        self.query = query
        self.metrics = metrics

        self.judgements = load("csv", self.settings['judgement_path'])
        if self.judgements.get_status():
            self.judgements = self.judgements.get_data()
            self.judgements.columns = ['query', 'owner_user_id']
        else:
            print("An Error Occurred ... --TERMINATING")
            exit()

        lmbd = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,0.7, 0.8, 0.9, 1.0]
        for par in lmbd:
            a = load("csv", self.settings['result_path'] + "ca_" + "_".join(self.query.split(' ')) + "@" + str(par) + ".csv")
            if a.get_status():
                run = "ca_" + "_".join(self.query.split(' ')) + "@" + str(par)
                self.results[run] = a.get_data()
                self.results[run]['rel'] = 0
                self.results[run]['precision'] = 0.0
                self.results[run]['recall'] = 0.0
                self.results[run].astype({'precision': float, 'recall': float})
            else:
                print("An Error Occurred ... --TERMINATING")
                exit()

        self.rel_ca_list = self.judgements.query('query == "' + self.query + '"')['owner_user_id'].tolist()
        self.rel_count = len(self.rel_ca_list)

        for par in lmbd:
            run = "ca_" + "_".join(self.query.split(' ')) + "@" + str(par)
            rel_ret_count = 0
            ret_count = self.results[run].shape[0]
            for index, row in self.results[run].iterrows():
                if self.expert_exists(row['owner_user_id']):
                    rel_ret_count += 1
                    self.results[run].at[index, "rel"] = 1
                    self.results[run].at[index, "precision"] = rel_ret_count / (index+1)
                    self.results[run].at[index, "recall"] = rel_ret_count / ret_count
            self.results_info[run] = {'AP': 0.0, 'p@k': 0.0, 'RR': 0.0, "rel_count": self.rel_count, "ret_count": ret_count, "RelRetCount": rel_ret_count, "Recall": rel_ret_count/self.rel_count, "Prec": rel_ret_count/ret_count}
        self.MAP()
        self.p_at_k()
        self.MRR()
        for q in self.results_info:
            print(q, self.results_info[q])


    def expert_exists(self, ca_no):
        return ca_no in self.rel_ca_list

    def MAP(self):
        runs = self.results.keys()
        for r in runs:
            rel = self.results[r].query("rel == 1")
            _map = np.mean(rel['precision'])
            self.results_info[r]['AP'] = _map

    def p_at_k(self):
        runs = self.results.keys()
        for r in runs:
            rel = self.results[r].head(self.k)
            _pAtk = np.sum(rel['rel']) / self.k
            self.results_info[r]['p@k'] = _pAtk

    def MRR(self):
        runs = self.results.keys()
        for r in runs:
            rel = self.results[r].head(self.rel_count)
            first_rel = rel.query('rel == 1')
            if first_rel.empty:
                _rr = 0
            else:
                _rr = 1 / (first_rel.head(1).index.tolist()[0] + 1)
            self.results_info[r]['RR'] = _rr