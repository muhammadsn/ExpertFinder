import pandas as pd


class DocModel:

    lmbd = 0.5

    def __init__(self, postings, collection, collection_model, doc_no, lmbd=0.5):
        self.postings = postings
        self.collection = collection
        self.cm = collection_model.drop_duplicates()
        self.doc_no = doc_no
        self.doc_word_list = []
        self.doc_unique_word_list = []
        self.doc_model = pd.DataFrame()
        self.lmbd = lmbd

        self.get_doc_model()


    def word_freq_in_doc(self, word):
        word_info = self.postings.query('word == "' + word + '"')
        for t in word_info['doc_n_freq'].tolist()[0]:
            if t[0] == str(self.doc_no):
                return t[1]
        else:
            return 0

    def doc_len(self):
        return len(self.doc_word_list)

    def MLE(self, word):
        return self.word_freq_in_doc(word=word) / self.doc_len()

    def estimate_doc_model(self):
        doc_info = self.collection.query('id == "' + str(self.doc_no) + '"')
        self.doc_word_list = doc_info['words'].tolist()[0][0] + doc_info['words'].tolist()[0][1]
        self.doc_unique_word_list = self.remove_duplicates(self.doc_word_list)
        rows = []
        for w in self.doc_unique_word_list:
            e = {'word': w, 'mle': self.MLE(w)}
            rows.append(e)
        model = pd.DataFrame(rows, columns=['word', 'mle'])
        self.doc_model = model

    # def calculate_doc_lang_model(self):
    #     df = self.doc_model.set_index('word').join(other=self.cm.set_index('word'), on='word', lsuffix="_doc", rsuffix="_col")
    #     df.drop_duplicates(inplace=True)
    #     df['smoothed'] = ((1-self.lmbd) * df['mle_doc'] + self.lmbd * df['mle_col'])
    #     self.doc_lang_model = df.reset_index()[['word', 'smoothed']]
    #     return self.doc_lang_model

    def get_doc_model(self):
        if self.doc_model.empty:
            self.estimate_doc_model()
        return self.doc_model

    def get_word_prob_in_model(self, word):
        p_c = self.cm.query('word == "' + word + '"')['mle'].tolist()
        p_d = self.doc_model.query('word == "' + word + '"')['mle'].tolist()
        p_c = p_c[0] if len(p_c) else 0
        p_d = p_d[0] if len(p_d) else 0
        return ((1-self.lmbd) * p_d) + (self.lmbd * p_c)

    @staticmethod
    def remove_duplicates(words):
        lst = list(dict.fromkeys(words))
        return lst