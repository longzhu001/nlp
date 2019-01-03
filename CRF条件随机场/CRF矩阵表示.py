# -*- coding: utf-8 -*-
"""

Aim:条件随机场(CRF, Conditional random field)的矩阵表示的python实现
"""
import numpy as np
class CCRF(object):
    '''
    实现条件随机场的矩阵表示
    '''
    
    def __init__(self, M):
        self.M = M #条件随机场的矩阵形式的存储体
        self.Z = None #规范化因子Z
        self.MP = []#矩阵乘积 Matrix Product
        
        self.work()
        return
        
    def work(self):
        print('work......')
		#np.full()函数可以生成初始化为指定值的数组 ,第一个参数是 列数   第二个参数是  生成值
        self.MP = np.full(shape=(np.shape(self.M[0])), fill_value=1.0)
        for i in range(np.shape(self.M)[0]):
            print('\nML=\n', self.MP)
            print('M%d=\n'%i, self.M[i])
			# np.dot矩阵乘法
            self.MP = np.dot(self.MP, self.M[i])
            print('dot=\n', self.MP)
    def ZValue(self):
        return self.MP[0,0]
        
def CCRF_manual():
    M1 = np.array([[0.5, 0.5],
                   [0,   0]])
    M2 = np.array([[0.3, 0.7],
                   [0.7, 0.3]])
    M3 = np.array([[0.5, 0.5],
                   [0.6, 0.4]])
    M4 = np.array([[1, 0],
                   [1, 0]])
    
    M = []
    M.append(M1)
    M.append(M2)
    M.append(M3)
    M.append(M4)
    M = np.array(M)
    print('CRF Matrix:\n', M)
    
    crf = CCRF(M)
    ret = crf.ZValue()
    print('从start到stop的规范化因子Z：', ret)
    return
    
if __name__=='__main__':
    CCRF_manual()
