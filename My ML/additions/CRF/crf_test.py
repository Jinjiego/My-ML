from collections import defaultdict
from additions.CRF.crf import *


def test():
     
     corps,tagids = read_corps('./additions/CRF/train.txt')
     featureTS,words2tagids = getfeatureTS(corps) #�õ��ܵ������б�featureTS
     K = np.shape(featureTS)[0] #�ܵ�������
     N = np.shape(corps)[0] #ѵ��������
     priorfeatureE = getpriorfeatureE(corps,featureTS) #������������������ֵ

     weights = np.array([1.0/K]*K)

     #postfeatureE = getpostfeatureE(weights,corps,featureTS,words2tagids)
     #liknegvalue = getliknegvalue(weights,corps,featureTS,words2tagids)
     weights,likelyfuncvalue = lbfgs(fun = getliknegvalue,gfun = getgradients,x0 = weights,corps = corps,
                                featureTS = featureTS,words2tagids = words2tagids,
                                priorfeatureE = priorfeatureE,m=10,maxk = 40)