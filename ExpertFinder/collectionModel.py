import pandas as pd


class CollectionModel:

    postings = pd.DataFrame()
    collection = pd.DataFrame()
    collection_len = 0
    collection_word_list = []
    collection_unique_word_list = []
    collection_model = pd.DataFrame()

    def __init__(self, postings, collection):
        self.postings = postings
        self.collection = collection

        self.get_collection_len()
        self.get_collection_model()

    def word_freq_in_collection(self, word):
        word_info = self.postings.query('word == "' + word + '"')
        word_freq_in_col = 0
        for t in word_info['doc_n_freq'].tolist()[0]:
            word_freq_in_col += t[1]
        return word_freq_in_col

    def get_collection_len(self):
        if self.collection_len == 0:
            for index, row in self.collection.iterrows():
                self.collection_word_list += row['words'][0]
                self.collection_word_list += row['words'][1]
            self.collection_len = len(self.collection_word_list)
            self.collection_unique_word_list = self.postings['word'].tolist()
        return self.collection_len

    def MLE(self, word):
        return self.word_freq_in_collection(word=word) / self.get_collection_len()

    def get_collection_model(self):
        if self.collection_model.empty:
            print(":: Calculating Collection Model ...", end="\t")
            rows = []
            # count = 1
            for w in self.collection_unique_word_list:
                e = {'word': w, 'mle': self.MLE(w)}
                rows.append(e)
                # print(count)
                # count += 1
            model = pd.DataFrame(rows, columns=['word', 'mle'])
            self.collection_model = model
            print("--DONE!")
        return self.collection_model
