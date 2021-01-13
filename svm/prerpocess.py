import random
import pickle
import argparse
import pandas as pd
from tqdm import tqdm
from scipy.sparse import csr_matrix

from datetime import datetime,timezone,timedelta
def timestamp(msg=""):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(f"{str(dt2)[:-13]}\t{msg}")

# Counter to vector
def counter_2_vector(doc_counter, query, lexicon):
    doc_vec = [0] * len(lexicon)
    for i, word in enumerate(lexicon):
        doc_vec[i] = doc_counter[word]
    for i, word in enumerate(query.split()):
        doc_vec[lexicon.index(word)] += 1
    return csr_matrix(doc_vec)

# sample training doc
"""
mode: train or test
df: queries, bm top 1000, bm scores dataframe
doc_dict: document counter dictionary
lexicon: list of string
"""
def get_svm_data(mode, df, doc_dict, lexicon):
    assert mode in ["train", "test"]
    q_id_list = df['query_id']
    q_list = df['query_text']
    if mode == "train":
        pos_doc_ids_list = df['pos_doc_ids']
    elif mode == "test":
        bm25_top1000_score_list = df['bm25_top1000_scores']
    bm25_top1000_list = df['bm25_top1000']
    svm_data = []

    for idx, q_id in tqdm(enumerate(q_id_list)):
        # query = q_list[idx]
        bm25_top1000 = bm25_top1000_list[idx].split()

        if mode == "train":
            # pos:neg = 1:1
            pos_count = 0
            pos_doc_ids = pos_doc_ids_list[idx].split()
            neg_doc = list(set(bm25_top1000) - set(pos_doc_ids))

            for r_doc in pos_doc_ids:
                pos_count += 1
                data_dict = dict()
                data_dict['q_id'] = q_id
                data_dict['query'] = q_list[idx]
                data_dict['d_id'] = r_doc
                data_dict['label'] = 1

                data_dict['doc_vec'] = counter_2_vector(doc_dict[r_doc], q_list[idx], lexicon)

                svm_data += [data_dict]
                
            neg_selected = set()
            for _ in range(pos_count):
                data_dict = dict()
                data_dict['q_id'] = q_id
                data_dict['query'] = q_list[idx]
                # sample negative document
                sampled_neg_doc = random.sample(neg_doc, 1)[0]
                while sampled_neg_doc in neg_selected:
                    sampled_neg_doc = random.sample(neg_doc, 1)[0]
                neg_selected.add(sampled_neg_doc)
                data_dict['d_id'] = sampled_neg_doc
                data_dict['label'] = 0

                data_dict['doc_vec'] = counter_2_vector(doc_dict[sampled_neg_doc], q_list[idx], lexicon)

                svm_data += [data_dict]

        elif mode  == "test":
            bm25_top1000_score = bm25_top1000_score_list[idx].split()
            for i, doc in enumerate(bm25_top1000):
                data_dict = dict()
                data_dict['q_id'] = q_id
                data_dict['query'] = q_list[idx]
                data_dict['doc_id'] = doc
                data_dict['doc_vec'] = counter_2_vector(doc_dict[doc], q_list[idx], lexicon)
                data_dict['bm_score'] = float(bm25_top1000_score[i])
                svm_data += [data_dict]
                
    return svm_data # List[Dict] = [{q_id, d_id, label, doc_vec}...] / [{q_id, d_id, doc_vec, bm_score}...]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", type=str, choices=["train", "test"], required=True)
    parser.add_argument("-df_path", type=str, required=True)
    parser.add_argument("-des_path", type=str, required=True)
    parser.add_argument("-doc_dict_path",type=str, required=True)
    parser.add_argument("-lexicon_path",type=str, required=True)
    args = parser.parse_args()

    train_df = pd.read_csv("dataset/train_queries.csv")
    test_df = pd.read_csv("dataset/test_queries.csv")
    query_words = []
    for q in train_df['query_text']:
        query_words += [w for w in q.split()]
    for q in test_df['query_text']:
        query_words += [w for w in q.split()]
    del train_df, test_df

    df = pd.read_csv(args.df_path)
    with open(args.doc_dict_path, "rb") as doc_fp:
        with open(args.lexicon_path, "rb") as lexicon_fp:
            with open(args.des_path, "wb") as fp:
                doc_dict = pickle.load(doc_fp)
                lexicon = pickle.load(lexicon_fp)
                lexicon += list(set(query_words))
                timestamp(f"each document convert to dimension {len(lexicon)} vector")
                with open("data/lexicon.pkl", "wb") as fp2:
                    pickle.dump(lexicon, fp2)
                pickle.dump(get_svm_data(args.mode, df, doc_dict, lexicon), fp)
                timestamp(f"result data saved at {args.des_path}")