# -*- coding: utf-8 -*-
__time__ = '2018/12/9 21:22'
__author__ = 'Mr.DONG'
__File__ = '全文检索系统.py'
__Software__ = 'PyCharm'

'''
实现一个简单的电影评论语料库的全文检索系统
'''

import nltk
import re


def raw(file):
    contents = open(file,'r').read()

    contents = re.sub(r'<.*?>', ' ', contents)

    contents = re.sub('\s+', ' ', contents)

    return contents


def snippet(doc, term):  # buggy

    text = ' ' * 30 + raw(doc) + ' ' * 30

    pos = text.index(term)

    return text[pos - 30:pos + 30]


print("Building Index...")

files = nltk.corpus.movie_reviews.abspaths()
# nltk.Index 利用词干提取器实现索引文本(concordance)
idx = nltk.Index((w, f) for f in files for w in raw(f))

query = ''

while query != "quit":

    query = input("query> ")

    if query in idx:

        for doc in idx[query]:
            print(snippet(doc, query))

    else:

        print("Not found")
