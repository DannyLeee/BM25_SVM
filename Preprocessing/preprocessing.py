import pandas as pd
import numpy as np
import pickle
from collections import Counter
path = 'dataset/'

doc_df = pd.read_csv(path + 'documents.csv')
doc_dict = dict()
# Fill nan(float) as none(str)
doc_df = doc_df.fillna('')

# Build remove terms
remove_title = ['[Text]', 'Language: <F P=105>', '</F>', 'Article Type:BFN', '<F P=106>', 'Article Type:CSO', '[Excerpt]', '[Editorial Report]', '[passage omitted]']
remove_term = ['.', '"', '--', '\'s', '<', '>', '[', ']', '`', ',', ':', '/', '\\', '{', '}']
# Build dictionary
for doc in doc_df.iloc:
    temp_str = doc['doc_text']
    for t in remove_title:
        if(t in temp_str):
            temp_str = temp_str.replace(t, '')
    for w in remove_term:
        if(w in temp_str):
            temp_str = temp_str.replace(w, '')
    temp = Counter(temp_str.split())
    doc_dict[doc['doc_id']] = temp

    # File output
pickle_out = open('./data/document.pickle', 'wb')
pickle.dump(doc_dict, pickle_out)
pickle_out.close()