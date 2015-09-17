# -*- coding: utf-8 -*-

'''
74.
73で学習したロジスティック回帰モデルを用い，
与えられた文の極性ラベル（正例なら"+1"，負例なら"-1"）と，
その予測確率を計算するプログラムを実装せよ．
75.
73で学習したロジスティック回帰モデルの中で，重みの高い素性トップ10と，重みの低い素性トップ10を確認せよ．
76.
学習データに対してロジスティック回帰モデルを適用し，正解のラベル，予測されたラベル，予測確率をタブ区切り形式で出力せよ．
77.
76の出力を受け取り，予測の正解率，正例に関する適合率，再現率，F1スコアを求めるプログラムを作成せよ．
'''


import argparse
import re
import numpy as np
import sklearn.cross_validation as crv
from sklearn.linear_model import LogisticRegression
from stemming.porter2 import stem

tp,tn,fp,fn = (0,0,0,0)


# 語幹抽出処理
def stemming(sentence):
        
    words = [stem(word.lower()) for word in re.sub("[\.\,\!\?;\:\(\)\[\]\'\"\(\)]$", '', sentence.rstrip()).split()] 
    return words

# 特徴量抽出処理(今回は最も単純と思われる、単語の頻度)
def get_feature(in_file):

    features = []
    
    with open(in_file,"r") as sentiment_file:
        for line in sentiment_file:
            label = line[:2]
            for word in stemming(line[3:]):
                features.append([label,word])
                
    features = np.array(features)       # 素性を持ったnumpy配列を作成。二次元配列。
    words = list(set(features[:,1]))    # 重複しているwordを削除する。python的。これはインデックスを指定して重複している数をカウントするため
    
    pos_vec = np.zeros(len(words))      # ポジティブベクトルの配列初期化
    neg_vec = np.zeros(len(words))      # ネガティブベクトルの配列初期化

    for feature in features:
        index = words.index(feature[1]) # indexメソッドで、対象の特徴語が、wordsリスト内のどの位置か調べる。
        if feature[0] == '-1':
            neg_vec[index] += 1         # pos_vecの指定のインデックスに、+1を加算
        else:
            pos_vec[index] += 1         # neg_vecの指定のインデックスに、+1を追加
                                                            
    return (features, pos_vec, neg_vec, words)  

# 学習
def train_logistic_regression(features, pos_vec, neg_vec):
    
    logit_model = LogisticRegression()  # 分類器のインスタンスを生成
    logit_model.fit([pos_vec, neg_vec],[1,-1]) # 訓練する。.fit(トレーニングベクトル,正解ベクトル)

    return (logit_model)


# 検証
def verification(label,predict):
    
    global tp,tn,fp,fn
    
    if label == predict and predict == 1 :
        tp += 1
    elif label == predict and predict == -1 :
        tn += 1
    elif label != predict and predict == 1 :
        fp += 1
    elif label != predict and predict == -1 :
        fn += 1
    else:
        print "error"
        exit()
        
# 検証結果表示        
def verification_result():
    
    accuracy = float(tp+tn)/(tp+tn+fn+fp)
    precision = float(tp)/(tp+fp)
    recall = float(tp)/(tp+fn)
    f = float((2*recall*precision))/(recall+precision)

    print 'Accuracy %f, precision %f, recall %f, F1 %f ' % (accuracy, precision, recall, f)

# スクリプト実行時    
if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", dest = "sentiment", default = "sentiment.txt")
    parser.add_argument("-v", "--verification", dest = "verification", default = "test_data.txt")
    args = parser.parse_args()
    
    
    
    print u"-------学習中----------"
    features, pos_vec, neg_vec, words= get_feature(args.sentiment) # データの特徴量算出
    features = np.array(features)          # 素性を持ったnumpy配列を作成。二次元配列。
# クロスバリデーション   
#    train_data, test_data, train_target, test_target = crv.train_test_split(features[:,1], features[:,0], test_size = 0.2 ,random_state = 0 )    
#    train = np.c_[train_target, train_data]
#    verification = np.c_[test_target,test_data]

# ロジスティック回帰モデルの作成、学習を行う
# 戻り値は単語とモデル   
    logit_train_model = train_logistic_regression(features, pos_vec, neg_vec)

# 以下モデルの検証用
    with open(args.verification,"r") as verification_file:
        for sentence in verification_file:
            label = sentence[:2]
            sentence = sentence[3:]
            input_vec = np.zeros(len(words))
            
            for word in stemming(sentence):
                try:
                    index = words.index(word)
                    input_vec[index] += 1
                except:
                    continue
                
#            print "正解ラベル: ", int(label) , "予測ラベル: ", int(logit_train_model.predict(input_vec)) , "予測確率: " , logit_train_model.predict_proba(input_vec)
            print u"正解ラベル: ", int(label) , u"予測ラベル: ", int(logit_train_model.predict(input_vec))             
            verification( int(label),int(logit_train_model.predict(input_vec)) )
            
        verification_result()            

