# -*- coding: utf-8 -*-
__time__ = '2018/11/26 21:33'
__author__ = 'Mr.DONG'
__File__ = 'jieba分词1.py'
__Software__ = 'PyCharm'


from nltk.tokenize import WordPunctTokenizer
import nltk
import jieba
from sklearn.feature_extraction.text import CountVectorizer
#or TfidfVectorizer 特征数值计算的常见方法
import scipy as sp

print ('test OK')


tokenizer = WordPunctTokenizer()
summaryList = []
file=open("./para.txt")
paras=file.readlines()
words=""
for para in paras:
    print (para)
    seg_list = list(jieba.cut(para, cut_all=False))
    words +=" ".join(seg_list)
    summaryList.insert(0," ".join(seg_list))
#para='I like eat apple because apple is red but because I love fruit'
#统计词频
sentences = tokenizer.tokenize(words)#此处将para转为list
#print sentences
wordFreq=nltk.FreqDist(sentences)
print (str(wordFreq.keys()))
#print dir(wordFreq)
print (str(summaryList))
#转换为词袋
vectorizer = CountVectorizer(min_df=0,max_df=20)
#summaryList 是一个列表，每一个元素是一个句子 词与词之间使用空格分开，默认不会处理单个词（即一个汉字的就会忽略）
#可以通过修改vectorizer的正则表达式，解决不处理单个字的问题
#vectorizer.token_pattern='(?u)\\b\\w+\\b'
X = vectorizer.fit_transform(summaryList)
print (str(vectorizer.get_feature_names()))
print (X.shape)
nums,features=X.shape   #帖子数量和词袋中的词数

#计算欧式距离
def dist_raw(v1,v2):
    delta=v1-v2
    return sp.linalg.norm(delta.toarray())

#测试
new_para='夏季新款清新碎花雪纺连衣裙，收腰显瘦设计；小V领、小碎花、荷叶袖、荷叶边的结合使得这款连衣裙更显精致，清新且显气质。'
new_para_list=" ".join(list(jieba.cut(new_para, cut_all=False)))
#new_para_list 是一个句子，词之间使用空格分开
new_vec=vectorizer.transform([new_para_list])


minDis = 9999
title=""
for i in range(0,nums):
    para = summaryList[i]
    para_vec=X.getrow(i)
    d=dist_raw(new_vec,para_vec)

    if(minDis > d):
        minDis = d
        title = para
print (title," = ",d)
print (new_para_list)
print (title)