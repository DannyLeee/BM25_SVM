import joblib
import pickle
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy import sparse
from sklearn.svm import SVC
from ml_metrics import mapk
from scipy.sparse import csr_matrix

from datetime import datetime,timezone,timedelta
def timestamp(msg=""):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(f"{str(dt2)[:-13]}\t{msg}")

def train(svm_data, filename):
    X = sparse.vstack([data['doc_vec'] for data in svm_data])
    y = [data['label'] for data in svm_data]

    timestamp("start training")
    clf = SVC()
    clf.fit(X, y)
    timestamp("training done")

    joblib.dump(clf, filename)
    timestamp(f"model saved at {filename}")

    return

def test(svm_data, model_path, test_df):
    clf = joblib.load(model_path)
    test_data = sparse.vstack([data['doc_vec'] for data in svm_data])
    
    timestamp("start testing")
    predictions = clf.predict(test_data)
    timestamp("testing done")

    q_list = test_df['query_id']
    doc_list = test_df['bm25_top1000']
    doc_score = test_df['bm25_top1000_scores']

    with open("result.csv", 'w') as fp:
        fp.write("query_id,ranked_doc_ids\n")
        for i, q_id in tqdm(enumerate(q_list)):
            d_list = doc_list[i].split()
            fp.write(str(q_id)+',')
            bm_score = np.array([float(s) for s in doc_score[i].split()])
            svm_score = []
            for j in range(1000):
                svm_score += [predictions[i*1000+j]]
            svm_score = np.array(svm_score)
            score = bm_score + 50*svm_score
            sortidx = np.argsort(score)
            sortidx = np.flip(sortidx)
            
            for idx in sortidx:
                fp.write(d_list[idx]+' ')
            fp.write("\n")
    timestamp("output saved at result.csv")

    return

def eval(ans_df, predict_df):
    actual = [doc_ids.split() for doc_ids in ans_df['RetrievedDocuments']]
    predicted = [doc_ids.split() for doc_ids in predict_df['ranked_doc_ids']]
    return mapk(actual, predicted, k=1000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-mode", choices=["train", "test", "eval"], type=str, required=True)
    parser.add_argument("-svm_data_path", type=str, required=True)
    parser.add_argument("-model_path", type=str, required=True, help="the model path to save or load (.pkl)")
    parser.add_argument("-df_path", type=str, help="test dataframe (.csv)")
    args = parser.parse_args()

    with open(args.svm_data_path, "rb") as fp:
        svm_data = pickle.load(fp)

    if args.mode == "train":
        train(svm_data, args.model_path)
    elif args.mode == "test":
        test_df = pd.read_csv(args.df_path)
        ans_df = pd.read_csv("dataset/FinalProjectTestSet/answer.txt")
        ans_df = ans_df.fillna("")
        test(svm_data, args.model_path, test_df)
        predict_df = pd.read_csv("result.csv")
        print(f"map@1000: {eval(ans_df, predict_df)}")        