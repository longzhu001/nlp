# -*- coding: utf-8 -*-
from tensorflow.python.keras import Input
from tensorflow.python.keras.engine.training import Model
from tensorflow.python.keras.layers import embeddings, LSTM, RepeatVector, Add, Dense
from tensorflow.python.layers.core import Dropout

__time__ = '2018/12/13 21:54'
__author__ = 'Mr.DONG'
__File__ = 'LSTM实现简单的问答系统.py'
__Software__ = 'PyCharm'


'''
数据预处理 
对文本数据进行向量化，word2vector
对文本数据 Tokenize，因为本数据集为英文，分词可直接用空格，如果数据集为中文，需要利用结巴或者其他分词器进行分词
'''
def tokenize(data):
    import re
    # ‘\W’ 匹配所有的字母数字下划线以外的字符
    return [x.strip() for x in re.split(r"(\W+)?", data) if x.strip()]

# 解析对话文本
# parse_dialog 将所有的对话进行解析，返回tokenize后的(对话,问题,答案)
# 如果 only_supporting为真表明只返回含有答案的对话
def parse_dialog(lines, only_supporting = False):
    data = []
    dialog = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ',1)
        nid = int(nid)
        # 标号为1表示新的一段文本的开始，重新记录
        if nid == 1:
            dialog = []
        #含有tab键的说明就是问题，将问题，答案和答案的索引分割开
        if '\t' in line:
            ques, ans, data_idx = line.split('\t')
            ques = tokenize(ques)
            substory = None
            if only_supporting :
                data_idx = list(map(int,data_idx))
                substory = [dialog[ i-1 ] for i in data_idx.split()]
            else:
                substory = [x for x in dialog]
            data.append((substory ,ques, ans))
        else:
            # 不含有tab键的就是对话，tokenize后加入dialog的list
            line = tokenize(line)
            dialog.append(line)
    return data

# 获得每个对话文本，将tokenize后的每个对话文本放在一个列表中。将（对话，问题，答案）组成相对应的tuple存储。
#这里的maxlen是控制文本最大长度的，可以利用分位数找出覆盖90%数据的长度，令其为maxlen。


# 否则序列长度太长，训练时内存不够。
from functools import reduce
def get_dialog(f, only_supporting = False, max_length = None):
#将对话完整的提取出来
    data = parse_dialog(f.readlines(),only_supporting = only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(dialog), ques, ans) for (dialog, ques, ans) in data if not max_length or len(flatten(dialog))<max_length]
    return data

def vectorize_dialog(data,wd_idx, dialog_maxlen, ques_maxlen):

#向量化,返回对应词表的索引号

    import numpy as np
    from keras.preprocessing.sequence import pad_sequences
    dialog_vec = []
    ques_vec = []
    ans_vec = []
    for dialog, ques, ans in data:
        dialog_idx = [wd_idx[w] for w in dialog]
        ques_idx = [wd_idx[w] for w in ques]

        ans_zero = np.zeros(len(wd_idx) + 1)
        ans_zero[wd_idx[ans] ] = 1

        dialog_vec.append(dialog_idx)
        ques_vec.append(ques_idx)
        ans_vec.append(ans_zero)

    #序列长度归一化，分别找出对话，问题和答案的最长长度，然后相对应的对数据进行padding。
    return pad_sequences(dialog_vec, maxlen = dialog_maxlen),\
            pad_sequences(ques_vec, maxlen = ques_maxlen),\
            np.array(ans_vec)
#准备数据
train_tar = tar.extractfile(data_path.format('train'))
test_tar = tar.extractfile(data_path.format('test'))
train = get_dialog(train_tar)
test = get_dialog(test_tar)

# 建立词表。词表就是文本中所有出现过的单词组成的词表。
lexicon = set()
for dialog, ques, ans in train + test:
    lexicon |= set(dialog + ques + [ans])
lexicon = sorted(lexicon)
lexicon_size = len(lexicon)+1

#word2vec，并求出对话集和问题集的最大长度，padding时用。
wd_idx = dict((wd, idx+1) for idx, wd in enumerate(lexicon))
dialog_maxlen = max(map(len,(x for x, _, _ in train + test )))
ques_maxlen =  max(map(len,(x for _, x, _ in train + test )))
#计算分位数，在get_dialog函数中传参给max_len
dia_80 = np.percentile(map(len,(x for x, _, _ in train + test )),80)

# 对训练集和测试集，进行word2vec
dialog_train, ques_train, ans_train = vectorize_dialog(train, wd_idx, dialog_maxlen, ques_maxlen)
dialog_test, ques_test, ans_test = vectorize_dialog(test, wd_idx, dialog_maxlen, ques_maxlen)

import numpy as np

#对话集 构建网络—— embedding + dropout
dialog = Input(shape = (dialog_maxlen, ),dtype='int32')
encodeed_dialog = embeddings.Embedding(lexicon_size, embedding_out)(dialog)
encodeed_dialog = Dropout(0.3)(encodeed_dialog)

#问题集 embedding + dropout + lstm
question = Input(shape = (ques_maxlen,),dtype= 'int32')
encodeed_ques = embeddings.Embedding(lexicon_size, embedding_out)(question)
encodeed_ques = Dropout(0.3)(encodeed_ques)
encodeed_ques = LSTM(units = lstm_out)(encodeed_ques)
encodeed_ques = RepeatVector(dialog_maxlen)(encodeed_ques)

# merge 对话集和问题集的模型 merge后进行 lstm + dropout + dense
merged = Add()([encodeed_dialog, encodeed_ques])
merged = LSTM(units = lstm_out)(merged)
merged = Dropout(0.3)(merged)
preds = Dense(units = lexicon_size, activation = 'softmax')(merged)

model = Model([dialog, question], preds)

print('compiling........')
model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy']
               )

#训练
print('training.......')
model.fit([dialog_train, ques_train], ans_train,batch_size = batch_size,epochs = epochs,verbose = 1,alidation_split = 0.1)
loss , accu = model.evaluate([dialog_test, ques_test], ans_test,
                             verbose= 1,
                             batch_size = batch_size)
print('%s: %.4f \n %s: %.4f' % ('loss', loss, 'accu', accu))

pre = model.predict([dialog_test, ques_test],
              batch_size = batch_size,
              verbose = 1)
#输出测试过程
def get_key(dic,value):
    return [k for k,v in wd_idx.items() if v == value]
import numpy as np
a = pre[0].tolist()
a.index(max(a))
for i in range(len(dialog_test)):
    ques = []
    lis_dia = list(map(lambda x : get_key(wd_idx,x), dialog_test[i]))
    dialog = reduce(lambda x,y :x+' '+y ,(reduce(lambda x, y: x+y,lis_dia)))

    lis_ques = (map(lambda x : get_key(wd_idx,x), ques_test[i]))
    ques = reduce(lambda x,y :x+' '+y,(reduce(lambda x, y: x+y,lis_ques)))

    ans_idx = np.argmax(ans_test[i])
    pre_idx = np.argmax(pre[i])
    print('%s\n %s ' % ('dialog',dialog))
    print('%s\n %s ' % ('question',ques))
    print('%s\n %s ' % ('right_answer',get_key( wd_idx, ans_idx)))
    print('%s\n %s\n' % ('pred',get_key(wd_idx, pre_idx)))

