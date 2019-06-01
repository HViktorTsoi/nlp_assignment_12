import pandas as pd
import numpy as np
import re
import jieba


def load_dict():
    """
    读文件
    获取中文词典
    :return:
    """
    cn_dict = pd.read_csv('./dict.txt', sep=' ').values[:, 0]
    cn_dict = {word: True for word in cn_dict}
    print('分词表长度: ', len(cn_dict))
    return cn_dict


def if_contain(words):
    """
    判断当前词在词典中是否存在
    :param words:
    :return:
    """
    return True if words in cn_dict else False


def spl(sentence):
    """
    逆向最大匹配算法的主要实现部分
    从后向前切割字符串，直到切割出的子串与词典中的词匹配
    :param sentence:
    :return:
    """
    result = ''
    words = []
    # 预处理 去掉标点符号和空格等停用词
    pat = '[\u3002|\uff1f|\uff01|\uff0c|\u3001|\uff1b|\uff1a|\u201c|\u201d|\u2018|\u2019|\uff08|\uff09|\u300a|\u300b|\u3008|\u3009|\u3010|\u3011|\u300e|\u300f|\u300c|\u300d|\ufe43|\ufe44|\u3014|\u3015|\u2026|\u2014|\uff5e|\ufe4f|\uffe5|\s|0-9|A-Za-z]'
    sentence = re.sub(pat, '', sentence)
    print('预处理后的文本: ', sentence)
    while len(sentence) > 0:
        except_flag = False
        for i in range(len(sentence), 0, -1):
            temp = sentence[:i]  # 中文字符串切割方式
            flag = if_contain(temp)
            if flag:
                words.append(temp)
                sentence = sentence[i:]
                except_flag = True
                print(len(words), ': ', temp)
                break
        if not except_flag:
            # 判断当前字符串是否在词典中并不存在，若该字符串从头切割到尾都没有词典中的词则认为无法切割并且
            # 词典中不存在，此时直接将该词当成切割后的结果加入结果列表
            words.append(sentence)
            break
    for w in words:
        result += (w + '/')
    return result


def evaluate(result, text):
    GT = list(jieba.cut(text))
    result = result.split('/')
    print('GT: ', GT)
    TP, FP, TN, FN = (0,) * 4
    TP = sum(1 for word in result if word in GT)
    print(TP, len(result), len(GT))
    prec = TP / len(result)
    recall = TP / len(GT)
    f1 = prec * recall * 2 / (2 * (prec + recall))
    print('Precision: {:.5}%'.format(prec * 100))
    print('Recall: {:.5}%'.format(recall * 100))
    print('F1: {:.5}%'.format(f1 * 100))


if __name__ == "__main__":
    cn_dict = load_dict()
    text = open('./wc_dataset/test.txt').read()
    result = spl(text)
    print("分词结果为： ", result)
    evaluate(result, text)
