import pickle

import jieba
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def load_stop_word():
    stop_words = open('./cls_dataset/stop/stopword.txt').readlines()
    stop_words = [word.replace('\n', '') for word in stop_words]
    print(stop_words)
    return stop_words


def process_dir_dataset(dataset_dir, stop_words, save_file, cv=None, tf_transformer=None):
    dataset = []
    labels = []
    for cls_id, category_path in enumerate(os.listdir(dataset_dir)):
        category_path = os.path.join(dataset_dir, category_path)
        for text_file in os.listdir(category_path):
            try:
                print(cls_id, text_file)
                text = open(os.path.join(category_path, text_file), encoding='GBK').read()
                dataset.append(' '.join(jieba.cut(text)))
                labels.append(cls_id)
                # break
            except UnicodeDecodeError:
                # 处理文件编码错误的情况
                print(category_path, text_file)
    # 词频统计
    # 测试集需要使用训练集的transformer
    if not cv:
        cv = CountVectorizer(stop_words=stop_words)
        cv = cv.fit(dataset)
    word_counts = cv.transform(dataset)
    print(word_counts.shape)
    # TF IDF
    # 测试集需要使用训练集的transformer
    if not tf_transformer:
        tf_transformer = TfidfTransformer(use_idf=False)
        tf_transformer = tf_transformer.fit(word_counts)
    tf_idf = tf_transformer.transform(word_counts)
    print(tf_idf.shape)
    # 生成并保存数据集
    pickle.dump({
        'data': tf_idf.toarray(),
        'label': np.array(labels)
    },
        open(save_file, 'wb')
    )
    return cv, tf_transformer


if __name__ == '__main__':
    # 处理训练集
    cv, tf_transformer = process_dir_dataset(
        dataset_dir='./cls_dataset/data',
        save_file='./train.pkl',
        stop_words=load_stop_word(),
    )
    # 处理测试集
    process_dir_dataset(
        dataset_dir='./cls_dataset/test1',
        save_file='./test.pkl',
        stop_words=load_stop_word(),
        cv=cv, tf_transformer=tf_transformer
    )
    # 载入数据集
    train_dataset = pickle.load(open('./train.pkl', 'rb'))
    test_dataset = pickle.load(open('./test.pkl', 'rb'))
    X_train, y_train = train_dataset['data'], train_dataset['label']
    X_test, y_test = test_dataset['data'], test_dataset['label']
    # 训练
    classifier = MultinomialNB()
    classifier = classifier.fit(X_train, y_train)
    # 测试
    print('TESTING.......')
    pred = classifier.predict(X_test)
    print(pred)
    acc = metrics.accuracy_score(y_test, pred)
    recall = metrics.recall_score(y_test, pred, average='macro')
    print('Acc: {:.5}%'.format(acc * 100))
    print('Recall: {:.5}%'.format(recall * 100))
