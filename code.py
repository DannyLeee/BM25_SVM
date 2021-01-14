import pickle
import argparse
import pandas as pd
from Preprocessing.preprocessing import preprocess
from BM25.bm25 import get_feature
from svm.prerpocess import get_svm_data, get_q_words
from svm.model import train, test, eval

from datetime import datetime,timezone,timedelta
def timestamp(msg=""):
    dt1 = datetime.utcnow().replace(tzinfo=timezone.utc)
    dt2 = dt1.astimezone(timezone(timedelta(hours=8))) # 轉換時區 -> 東八區
    print(f"{str(dt2)[:-13]}\t{msg}")

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-mode", type=str, choices=["train_scratch", "train", "test", "preprocess", "get_feature", "get_svm_data"], required=True)
parser.add_argument("-doc_dict_path", type=str, default="data/document.pkl", help="needed mode: except test")
parser.add_argument("-hidden_size", type=int, default=50, help="feature words size; needed mode: get_feature")
# bm25
parser.add_argument("-bm_scratch", type=int, help="selete feature words from scratch or not; needed mode: train, get_feature")
parser.add_argument("-feature_word_path", type=str, help="file path to save feature_words after selecting; needed mode: get_feature")
# get svm data
parser.add_argument("-df_path", type=str, help="preprocess/testing dataframe path (.csv); needed mode:train_scratch, test, get_svm_data")
parser.add_argument("-svm_data_path", type=str, help="file path to save svm_data after preprocess or load when testing; needed mode: test, get_svm_data")
parser.add_argument("-lexicon_path", type=str, help="feature_words file path; needed mode: get_svm_data")
# model
parser.add_argument("-model_mode", type=str, choices=["train", "test"])
parser.add_argument("-model_path", type=str, help="the model path to save or load (.pkl); needed mode: train, test")
args = parser.parse_args()


# argment condition check
if args.mode == "get_svm_data":
    if args.model_mode == None:
        parser.error(f"the following arguments are required: -model_mode in mode `{args.mode}`")
if args.mode == "train_scratch" or args.mode == "test" or args.mode == "get_svm_data":
    if args.df_path == None:
        parser.error(f"the following arguments are required: -df_path in mode `{args.mode}`")
if args.mode == "train" or args.mode == "test":
    if args.model_path is None:
        parser.error(f"the following arguments are required: -model_path in mode `{args.mode}`")
if args.mode == "get_feature":
    if args.feature_word_path is None:
        parser.error(f"the following arguments are required: -feature_word_path in mode `{args.mode}`")
            


if args.mode == "train_scratch":
    # raw_data preprocess to Counter for each document
    timestamp("preprocess start")
    doc_dict = preprocess(args) # {"doc_id": Counter of document}

    # bm25 select feature words
    timestamp("select feature start")
    feature_words = get_feature(args, doc_dict) # ["words"...]

    # add query words to lexicon
    lexicon = feature_words
    lexicon += list(set(get_q_words()))
    lexicon = list(set(lexicon))
    df = pd.read_csv(args.df_path)
    
    # vectorization to svm data
    timestamp(f"each document convert to dimension {len(lexicon)}'s vector")
    svm_data, svm_data_nq = get_svm_data(args.mode, df, doc_dict, lexicon)

    # model training
    train(svm_data, args.model_path)
    train(svm_data_nq, args.model_path[:-4]+"_nq.pkl")

elif args.mode=="train":
    with open(args.svm_data_path, "rb") as fp:
        svm_data = pickle.load(fp)
    with open(args.svm_data_path[:-4]+"_nq.pkl", "rb") as fp:
        svm_data_nq = pickle.load(fp)

    # model training
    train(svm_data, args.model_path)
    train(svm_data_nq, args.model_path[:-4]+"_nq.pkl")

elif args.mode == "test":
    with open(args.svm_data_path, "rb") as fp:
        svm_data = pickle.load(fp)
    with open(args.svm_data_path[:-4]+"_nq.pkl", "rb") as fp:
        svm_data_nq = pickle.load(fp)
    test_df = pd.read_csv(args.df_path)
    ans_df = pd.read_csv("dataset/FinalProjectTestSet/answer.txt")
    ans_df = ans_df.fillna("")
    test(svm_data, args.model_path, test_df)
    predict_df = pd.read_csv("result.csv")
    print(f"have query map@1000: {eval(ans_df, predict_df)}")

    with open(args.svm_data_path[:-4]+"_nq.pkl", "rb") as fp:
        svm_data_nq = pickle.load(fp)
    test(svm_data_nq, args.model_path[:-4]+"_nq.pkl", test_df)
    predict_df = pd.read_csv("result.csv")
    print(f"non query map@1000: {eval(ans_df, predict_df)}")

elif args.mode == "preprocess":
    timestamp("preprocess start")
    preprocess(args) # {"doc_id": Counter of document}

elif args.mode == "get_feature":
    # bm25 select feature words
    timestamp("select feature start")
    feature_words = get_feature(args) # ["words"...]
    print('Save feature_words')
    with open(args.feature_word_path, 'wb') as f:
        pickle.dump(feature_words, f)

elif args.mode == "get_svm_data":
    with open(args.doc_dict_path, "rb") as doc_fp:
        doc_dict = pickle.load(doc_fp)
    with open(args.lexicon_path, "rb") as lexicon_fp:
        lexicon = pickle.load(lexicon_fp)
    
    # add query words to lexicon
    lexicon += list(set(get_q_words()))
    lexicon = list(set(lexicon))
    lexicon.sort()
    with open("data/lexicon.pkl", "wb") as fp:
        pickle.dump(lexicon, fp)
    df = pd.read_csv(args.df_path)
    
    # vectorization to svm data
    timestamp(f"each document convert to dimension {len(lexicon)}'s vector")
    svm_data, svm_data_nq = get_svm_data(args.model_mode, df, doc_dict, lexicon)

    with open(args.svm_data_path, "wb") as fp:
        pickle.dump(svm_data, fp)
    with open(args.svm_data_path[:-4]+"_nq.pkl", "wb") as fp:
        pickle.dump(svm_data_nq, fp)
    timestamp(f"result data saved at {args.svm_data_path} and {args.svm_data_path[:-4]+'_nq.pkl'}")