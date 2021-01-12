import pandas as pd
import numpy as np
import pickle
from collections import Counter
import nltk
import ssl
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context
nltk.download('stopwords')
from nltk.corpus import stopwords
path = './dataset/'

doc_df = pd.read_csv(path + 'documents.csv')
doc_dict = dict()
# Fill nan(float) as none(str)
doc_df = doc_df.fillna('')
Languages = list(stopwords.fileids())

# Build remove terms
remove_title = ['[Text]', 'Language:', '<F P=105>', '</F>', 'Article Type:BFN', '<F P=106>', 'Article Type:CSO', '[Excerpt]', '[Editorial Report]', '[passage omitted]', 'ONLY <F P=103>', '<F P=104>']
remove_term = ['.', '"', '--', '\'s', '<', '>', '[', ']', '`', ',', ':', '/', '\\', '{', '}', '-', '(', ')']
my_stopwords = ['mr', 'he\'d', 'also', 'every', 'would', 'without', 'per', 'yesterday', 'however', 'could', 'since', 'many', 'must', 'well', 'still', 'today', 'people', 'next']

print('Stopwords removing processing\n')
# Build dictionary
for doc in doc_df.iloc:
    temp_str = doc['doc_text']
    # Choosing the fileids of stopwords, initial:english
    Lang_tmp = 'english'
    Lang_flag = False
    if('Language: <F P=105>' in temp_str):
        Lang_flag = True
        Lang_tmp = temp_str.split('Language: <F P=105>')[1].split()[0]
        if(not(Lang_tmp.lower() in Languages)):
            Lang_flag = False
    # Removing meaningless words
    for t in remove_title:
        if(t in temp_str):
            temp_str = temp_str.replace(t, '')
    for w in remove_term:
        if(w in temp_str):
            temp_str = temp_str.replace(w, '')
    # Removing stopwords
    temp = temp_str.split()
    tmp_len = len(temp)
    for t in range(tmp_len):
        temp[t] = temp[t].lower()
    temp = Counter(temp)
    for m in my_stopwords: # My stopwords set for all doc
        if(m in temp):
            del temp[m]
    for s in stopwords.words('english'): # english stopwords for all doc
        if(s in temp):
            del temp[s]
    if(Lang_flag and Lang_tmp != 'English'):
        for s in stopwords.words(Lang_tmp.lower()):
            if(s in temp):
                del temp[s]
    # Save to dict    
    doc_dict[doc['doc_id']] = temp

print('.pkl file output\n')
# File output
pickle_out = open('./data/document.pkl', 'wb')
pickle.dump(doc_dict, pickle_out)
pickle_out.close()

print('Down\n')