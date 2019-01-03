# -*- coding: utf-8 -*-
import re

import nltk

__time__ = '2018/12/9 21:38'
__author__ = 'Mr.DONG'
__File__ = 'aa.py'
__Software__ = 'PyCharm'



def raw(file):
    with open(file) as f:
        contents=f.read()
    contents = list(contents)
    contents = re.sub(r'<.*?>', ' ', contents)

    contents = re.sub('\s+', ' ', contents)

    return contents


files = nltk.corpus.movie_reviews.abspaths()
for f in files:
    print(f)
    content = raw(f)
    print(content)