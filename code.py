import argparse
import pandas as pd
from Preprocessing.preprocessing import preprocess
from BM25.bm25 import get_feature
from svm.prerpocess import get_svm_data
from svm.model import train, test, eval

from datetime import datetime,timezone,timedelta
def timestamp(msg=""):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(f"{str(dt2)[:-13]}\t{msg}")

parser = argparse.ArgumentParser()
parser.add_argument("-mode", type=str, choices=["preprocess", "get_feture", "get_svm_data", "model", "scratch"], default="scratch")
parser.add_argument("-doc_dict_path", type=str, default="data/document.pkl")
parser.add_argument("-hidden_size", type=int, default=50)
# bm25
parser.add_argument("-bm_scratch", type=int)
# get svm data
parser.add_argument("-df_path", type=str, help="preprocess/testing dataframe path (.csv)")
# model
parser.add_argument("-model_mode", type=str, choices=["train", "test", "eval"])
parser.add_argument("-model_path", type=str)
args = parser.parse_args()

print(args)

# argment condition check
if args.mode == "scratch" or args.mode == "model" or args.mode == "get_svm_data":
    if args.model_mode == None:
        parser.error(f"the following arguments are required: -model_mode in mode \`{args.mode}\`")
if args.mode == "scratch" or (args.mode == "model" and args.model_mode == "test") or args.mode == "get_svm_data":
    if args.df_path == None:
        parser.error(f"the following arguments are required: -df_path in mode \`{args.mode}\`")
if args.mode == "scratch" or args.mode == "model":
    if args.model_mode == "train" or args.model_mode == "test":
        if args.model_path is None:
            parser.error(f"the following arguments are required: -model_path in model_mode \`{args.model_mode}\`")
            


if args.mode == "scratch":
    # raw_data preprocess to Counter for each document
    timestamp("preprocess start")
    doc_dict = preprocess(args) # {"doc_id": Counter of document}

    # bm25 select feature words
    timestamp("select feature start")
    feature_words = get_feature(args, doc_dict) # ["words"...]

    # add query words to lexicon
    train_df = pd.read_csv("dataset/train_queries.csv")
    test_df = pd.read_csv("dataset/test_queries.csv")
    query_words = []
    for q in train_df['query_text']:
        query_words += [w for w in q.split()]
    for q in test_df['query_text']:
        query_words += [w for w in q.split()]
    del train_df, test_df
    lexicon = feature_words
    lexicon += list(set(query_words))
    df = pd.read_csv(args.df_path)
    
    # vectorization to svm data
    timestamp(f"each document convert to dimension {len(lexicon)} vector")
    svm_data = get_svm_data(args.model_mode, df, doc_dict, lexicon)
    
    # model training
    train(svm_data, args.model_path)   

# elif args.mode == "preprocess":
#     #
# elif args.mode == "get_feture":
#     #
# elif args.mode == "get_svm_data":
#     #
# elif args.mode == "model":
#     #