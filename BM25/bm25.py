import pickle
from tqdm import tqdm
from collections import Counter
from scipy.sparse import csr_matrix
import numpy as np
from numba import njit



def term_frequency(index_term_dict, document):
    data = []
    row = []
    col = []
    r = -1
    for doc in tqdm(document.keys(), desc='Term Frequency'):
        r += 1
        for term in document[doc].keys():
            if term in index_term_dict:
                c = index_term_dict[term]
                row.append(r)
                col.append(c)
                data.append(document[doc][term])
    data = np.array(data)
    row = np.array(row)
    col = np.array(col)
    tf_matrix = csr_matrix((data, (row, col)), shape=(len(document), len(index_term_dict)), dtype=np.float)
    return tf_matrix


def document_frequency(doc_tf_matrix):
    doc_tf_col_counter = Counter(doc_tf_matrix.tocoo().col)
    df_list = []
    for i in tqdm(range(len(doc_tf_col_counter)), desc='Document Frequency'):
        df_list.append(doc_tf_col_counter[i])
    df_matrix = np.array(df_list)
    return df_matrix


@njit
def _term_bm25_weight(tf_data, tf_row, tf_col, len_term, idf_data, bm_doc_len_data, k1):
    term_bm25_weight = np.zeros(len_term)
    for term_index, doc_index, tf in zip(tf_row, tf_col, tf_data):
        term_bm25_weight[term_index] += (k1 + 1) * tf / (tf + bm_doc_len_data[doc_index]) * idf_data[term_index]
    return term_bm25_weight



# Reading document pickle file
print('Reading document pickle file')
with open('./data/document.pickle', 'rb') as f:
    document = pickle.load(f)

# Get index_term
index_term = set()
for doc in tqdm(document.keys(), desc='Get index_term'):
    index_term |= set(document[doc].keys())
index_term = list(index_term)

# k:term , v:term_index
index_term_dict = {k: v for v, k in enumerate(index_term)}

# k:doc_id , v:doc_index
doc_dict = {k: v for v, k in enumerate(list(document.keys()))}

# TF, DF, IDF
tf_matrix = term_frequency(index_term_dict, document)
df_matrix = document_frequency(tf_matrix)
idf_matrix = np.log((len(document) - df_matrix + 0.5) / (df_matrix + 0.5))

# BM25
print('Create BM25 matrix')
k1 = 2.5
b = 0.8
doc_len = [sum(document[doc].values()) for doc in document.keys()]
avg_doclen = sum(doc_len) / len(document)
tf_matrix_T = tf_matrix.tocoo().transpose()

bm25_matrix = np.zeros(len(index_term))
v2 = np.array(doc_len)
v2 = k1 * ((1 - b) * b * (v2 / avg_doclen))
bm25_matrix = _term_bm25_weight(tf_matrix_T.data, tf_matrix_T.row, tf_matrix_T.col, len(index_term_dict), idf_matrix, v2, k1)

# Save BM25 matrix
print('Save BM25 matrix')
np.save('bm25_matrix.npy', bm25_matrix)

# Get feature_words
print('Get feature_words')
feature_amount = 50
feature_words = []
for term_index in np.flip(bm25_matrix.argsort())[:feature_amount]:
    feature_words.append(index_term[term_index])

print('Save feature_words')
with open('./data/feature_words.pickle', 'wb') as f:
    pickle.dump(feature_words, f)

print('Done')