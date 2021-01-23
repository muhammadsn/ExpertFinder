import xml.etree.ElementTree as et
import pandas as pd
from .textProcessor import TextProcessor as tp
from .fileHandler import Exporter as save
from .fileHandler import Importer as load

pd.set_option("display.max_rows", None, "display.max_columns", None)


class Indexer:

    settings = {
        'stemmer': "",
        'resource_path': "",
        'dataset_path': "",
        'data_path': "",
        'index_path': "",
    }

    dataset = pd.DataFrame()
    postings = pd.DataFrame()
    all_stop_words = []                             # all stop words in english --WITH DUPLICATES
    all_words = []                                  # all words in corpus without punctuation signs --WITH DUPLICATES
    all_discriminative_words = []                   # all discriminative words in corpus --WITH DUPLICATES
    all_unique_words = []                           # all words in corpus without punctuation signs
    all_unique_stop_words = []                      # all stop words in corpus without punctuation signs
    all_unique_discriminative_words = []            # all discriminative words in their stemmed format

    def __init__(self, resource_path, data_file_name, index_path, index_file_name, dataset_file_name, stemmer):

        self.settings['resource_path'] = resource_path
        self.settings['data_path'] = resource_path + data_file_name
        self.settings['index_path'] = index_path + index_file_name
        self.settings['dataset_path'] = resource_path + dataset_file_name
        self.settings['stemmer'] = stemmer

        data = load(_format="json", _path=self.settings['dataset_path'])

        if data.get_status():
            self.dataset = data.get_data()
            self.word_extraction()
        else:
            self.xml_data_parser()

        self.create_postings()

    def xml_data_parser(self):
        """
        Parses xml dataset of posts of Stack Over Flow
        :return:
        """
        print(":: Parsing Input Files ...", end="\t")
        xtree = et.parse(self.settings['data_path'])
        xroot = xtree.getroot()

        cols = ["id", "post_type_id", "parent_id", "creation_date", "score", "owner_user_id", "last_activity_date", "comment_count", "words", "body"]
        rows = []

        for node in xroot.findall('row'):
            s_ID = node.get("Id") if node is not None else None
            s_PostTypeId = node.get("PostTypeId") if node is not None else None
            s_ParentId = node.get("ParentId") if node is not None else None
            s_CreationDate = node.get("CreationDate") if node is not None else None
            s_Score = node.get("Score") if node is not None else None
            s_OwnerUserId = node.get("OwnerUserId") if node is not None else None
            s_LastActivityDate = node.get("LastActivityDate") if node is not None else None
            s_CommentCount = node.get("CommentCount") if node is not None else None
            t_Body = node.get("Body") if node is not None else None
            t_words = tp(t_Body, self.settings['stemmer'])
            s_Words = [t_words.get_words(), t_words.get_stop_words()]
            s_Body = " ".join(s_Words[0]) + " " + " ".join(s_Words[1])
            self.all_discriminative_words += s_Words[0]
            self.all_words += t_words.get_all_words()
            self.all_stop_words += s_Words[1]

            rows.append({"id": s_ID, "post_type_id": s_PostTypeId, "parent_id": s_ParentId,
                         "creation_date": s_CreationDate, "score": s_Score, "owner_user_id": s_OwnerUserId,
                         "last_activity_date": s_LastActivityDate, "comment_count": s_CommentCount, "words": s_Words, "body": s_Body})
        print("--DONE!")

        print(":: Creating Initial Dataset ...", end="\t")
        self.dataset = pd.DataFrame(rows, columns=cols)
        print("--DONE!")

        print(":: Saving Dataset Structure to File ...", end="\t")
        save(_data=self.dataset, _format="json", _path=self.settings['dataset_path'])
        print("--DONE!")

        return self.dataset

    def create_postings(self):

        print(":: Initializing Lexicon and Postings...")
        cols = ["word", "doc_n_freq"]
        rows = []

        self.all_unique_words = self.remove_duplicates(self.all_words)
        self.all_unique_stop_words = self.remove_duplicates(self.all_stop_words)
        self.all_unique_discriminative_words = self.remove_duplicates(self.all_discriminative_words)

        print(":: " + str(len(self.all_unique_discriminative_words) + len(self.all_unique_stop_words)) + " Words Found")

        count = 1

        for w in (self.all_unique_discriminative_words + self.all_unique_stop_words):
            rows.append({'word': w, 'doc_n_freq': self.word_frequency(w)})
            print(count)
            # print(rows[-1])

            if count % 10000 == 0 or count == (len(self.all_unique_discriminative_words) + len(self.all_unique_stop_words))-1:
                self.postings = pd.DataFrame(rows, columns=cols)
                print(":: [CHECKPOINT " + str(count/10000) + "] Saving Postings to File ...", end="\t")
                save(_data=self.postings, _format="json", _path=self.settings['index_path'] + str(count/10000)+".json")
                rows = []
                print("--DONE!")

            count += 1

        print(":: Creating Lexicon and Postings ...\t --DONE!")

    def word_frequency(self, word):
        appearance = []
        idx = self.dataset.index[self.dataset['body'].str.contains(word)].tolist()
        for i in idx:
            appearance.append((self.dataset.iloc[i]['id'], sum(word in s for s in self.dataset.iloc[i]['words'][0] + self.dataset.iloc[i]['words'][1])))
        return appearance

    def word_extraction(self):
        print(":: Initializing Word Lists...", end="\t")
        for index, row in self.dataset.iterrows():
            self.all_discriminative_words += row['words'][0]
            self.all_stop_words += row['words'][1]
        print("--DONE!")

    @staticmethod
    def remove_duplicates(words):
        lst = list(dict.fromkeys(words))
        return lst
