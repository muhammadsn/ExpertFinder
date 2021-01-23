import pandas as pd
import numpy as np
from .textProcessor import TextProcessor as tp
from .collectionModel import CollectionModel as cm
from .docModel import DocModel as dm
from .fileHandler import Importer as load
from .fileHandler import Exporter as save


pd.set_option("display.max_rows", None, "display.max_columns", None)


class Retriever:

    settings = {
        'stemmer': "",
        'resource_path': "",
        'dataset_path': "",
        'index_path': "",
        'result_path': "",
        'doc_ranking_path': ""
    }

    lmbd = 0
    req = 1000
    query = ""
    query_words = ""
    query_stop_words = ""
    postings = pd.DataFrame()
    collection = pd.DataFrame()
    collection_model = None
    ca_ranking = pd.DataFrame()

    def __init__(self,  resource_path, result_path, doc_ranking_path, index_path, index_file_name, dataset_file_name, stemmer, query, req_count, lmbd):
        self.settings['resource_path'] = resource_path
        self.settings['result_path'] = result_path
        self.settings['doc_ranking_path'] = doc_ranking_path
        self.settings['index_path'] = index_path + index_file_name
        self.settings['dataset_path'] = resource_path + dataset_file_name
        self.settings['stemmer'] = stemmer
        self.lmbd = lmbd
        self.query = query
        self.req = req_count

        self.load_index()
        self.load_collection()
        self.query_processor()
        print(self.query_words)
        print(self.query_stop_words)

        tmp = load(_format="json", _path=self.settings['resource_path']+"collectionModel.json")
        if tmp.get_status():
            self.collection_model = tmp.get_data()
        else:
            a = cm(self.postings, self.collection)
            self.collection_model = a.get_collection_model()
            save(self.collection_model, "json", self.settings['resource_path'] + "collectionModel.json")

        self.search()


    def load_index(self):
        index = load("json", self.settings['index_path'])
        self.postings = index.get_data()

    def load_collection(self):
        collection = load("json", self.settings['dataset_path'])
        collection = collection.get_data()
        collection.drop(columns=['body', 'last_activity_date', 'creation_date'], inplace=True)
        self.collection = collection

    def query_processor(self):
        q = tp(self.query, self.settings['stemmer'])
        self.query_words = q.get_words()
        self.query_stop_words = q.get_stop_words()

    def rel_doc_finder(self, word):
        doc_list = self.postings.query('word == "' + word + '"')
        doc_list = doc_list['doc_n_freq'].tolist()[0]
        return doc_list

    def search(self):
        print(":: Initializing Document retrieval for \"" + self.query + "\" for lambda = " + str(self.lmbd) + " ...")
        all_query_words = self.query_words + self.query_stop_words
        ranking = load("json", self.settings['doc_ranking_path'] + "_".join(all_query_words) + "@" + str(self.lmbd) + ".json")

        if ranking.get_status():
            ranking = ranking.get_data()
        else:
            print(":: Calculating Document Scores ...", end="\t")
            scores = {}
            for qw in all_query_words:
                nwq = all_query_words.count(qw)
                rel_doc_list = self.rel_doc_finder(qw)
                tot_count = len(rel_doc_list)
                count = 1
                for doc in rel_doc_list:
                    theta_d = dm(self.postings, self.collection, self.collection_model, doc[0], self.lmbd)
                    score = theta_d.get_word_prob_in_model(qw)
                    if doc[0] in scores:
                        scores[doc[0]] += nwq * np.log(score)
                    else:
                        scores[doc[0]] = nwq * np.log(score)
                    print(str(count) + " / " + str(tot_count) + "\t" + doc[0] + "\t" + str(scores[doc[0]]))
                    count += 1
            ranking = pd.DataFrame.from_dict(scores, orient='index').reset_index()
            ranking.columns = ['doc', 'score']
            print(":: Saving Document Ranking for \"" + self.query + "\" ...", end="\t")
            save(ranking, "json", self.settings['doc_ranking_path'] + "_".join(all_query_words) + "@" + str(self.lmbd) + ".json")
            print("--DONE!")

        self.ca_scorer(ranking)

    def ca_scorer(self, query_post_data):
        print(":: Performing Ranking Candidates for \"" + self.query + "\" for lambda = " + str(self.lmbd) + " ...", end="\t")
        post_ca_data = self.collection[['id', 'owner_user_id']]
        post_ca_data.columns = ['doc', 'owner_user_id']
        df = query_post_data.set_index('doc').join(post_ca_data.set_index('doc'), how='inner', on='doc')
        df.reset_index(inplace=True)
        ca_ranking = df.groupby(['owner_user_id']).sum()
        ca_ranking.sort_values(by=['score'], ascending=True, inplace=True)
        ca_ranking.reset_index(inplace=True)
        ca_ranking['query'] = self.query
        self.ca_ranking = ca_ranking[['query', 'owner_user_id', 'score']].head(self.req)
        print("--DONE!")
        print(":: Saving Candidate Ranking for \"" + self.query + "\" for lambda = " + str(self.lmbd) + " ...", end="\t")
        save(self.ca_ranking, "csv", self.settings['result_path'] + "ca_" + "_".join(self.query.split(" ")) + "@" + str(self.lmbd) + ".csv")
        print("--DONE!")