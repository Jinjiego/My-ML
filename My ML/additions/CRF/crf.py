#coding = utf-8
from collections import defaultdict
from additions.CRF.Logspace import Logspace

import numpy as np
def read_corps(corpsfile='testchunk.data'):
    #http://www.chokkan.org/software/crfsuite/tutorial.html,��ҳ����������ַ���������ݼ��������ݼ����ܴ�
    #http://blog.dpdearing.com/2011/12/opennlp-part-of-speech-pos-tags-penn-english-treebank/
    tagids = defaultdict(lambda: len(tagids))
    tagids["<S>"] = 0

    corps=[]
    onesentence = []
    words = [ "<S>" ]
    tags  = [   0   ]
    #wordnumcount = 0
    with open(corpsfile,'r') as f:   
        for line in f:
            if len(line)<=1:
                pass
            elif line != '. . O\n': 
                # '. . O\n'��ʾһ�仰��������һ�仰δ�����򽫸õ��ʼ����б�onesentence
                onesentence.append(line)
            else: #���һ�仰��������Ըþ仰�����г��ֵĵ��ʽ��д����������������б�corps               
                for texts in onesentence:
                    #wordnumcount += 1
                    w_t = texts.strip().split(" ")
                    #print w_t
                    try: 
                        #���ڱ�ʾ���ֵ��ַ����仯�϶࣬Ϊ�˼�������ţ����ｫ����������滻��
                        float(w_t[0].strip().replace(',',''));
                        #print w_t
                        words.append('#CD#')
                    except: 
                        words.append(w_t[0].lower()) 
                    #if w_t[1] in{ '``',',',"''",'$','#',')','('}:
                    #    print w_t
                    tags.append(tagids[w_t[1]])
                words.append("<S>") #words��һ�仰�ĵ�����ɵ��б�
                tags.append(0)      #tags��һ�仰�ı�ע��ɵ��б��뵥���б�wordsһһ��Ӧ
                if np.shape(words)[0] > 2: #�ų����վ���
                    corps.append((words,tags))

                #��onesentence��words��tags���³�ʼ��
                onesentence = []
                words = [ "<S>" ]
                tags  = [   0   ]
    #print 'һ�����ֵĵ��ʸ�����'+np.str(wordnumcount)
    #һ�����ֵĵ��ʸ�����40377
    return corps,tagids
def getfeatureTS(corps):
    featuresets = set() #�����ļ���
    featureT = [] #ת���������б������б�Ԫ��('T', 2, 3)��ʾ��״̬2ת������3
    featureS = [] #״̬�������б������б�Ԫ��('S','Confidence', 1)
    for corp in corps:
        for i in range(np.shape(corp[0])[0]):
            if corp[0][i] == '<S>':
                continue
            if ('S',corp[0][i],corp[1][i]) not in featuresets:
                featuresets.add(('S',corp[0][i],corp[1][i]))
                featureS.append(('S',corp[0][i],corp[1][i]))
            if corp[0][i-1] != '<S>':
                if ('T',corp[1][i-1],corp[1][i]) not in featuresets:
                    featuresets.add(('T',corp[1][i-1],corp[1][i]))
                    featureT.append(('T',corp[1][i-1],corp[1][i]))
    featureTS = featureT+featureS
    words2tagids = words2tagidfromfeatureS(featureS)
    return featureTS,words2tagids
def getpriorfeatureE(corps,featureTS):
    #����������������ֵ
    N = np.shape(corps)[0] #ѵ��������
    K = np.shape(featureTS)[0] #������
    priorfeatureE = np.zeros(K) 

    for corp in corps: 
        for i in range(np.shape(corp[0])[0]):
            if corp[0][i] == '<S>':
                continue 
            try:
                idex = featureTS.index(('S', corp[0][i], corp[1][i]))
                priorfeatureE[idex] += 1.0
            except:
                pass
            try:
                idex = featureTS.index(('T', corp[1][i-1], corp[1][i]))
                priorfeatureE[idex] += 1.0
            except:
                pass
    priorfeatureE /=N
    #plt.plot(priorfeatureE) 
    #����������������ֵ���Կ���������ת������(�Ӻ�����0��ʼ)����״̬����(�Ӻ�����318��ʼ)��
    #�ȱ���¼����������ֵԽ��
    return priorfeatureE
def words2tagidfromfeatureS(featureS):
    #ͳ�����е��ʷֱ��Ӧ�Ĵ����б�
    words2tagids = {}
    for feature in featureS:
        word = feature[1]
        state = feature[2]
        if word in words2tagids:
            words2tagids[word].append(state)
        else:
            words2tagids[word] = [state]

    #lennums�б�ͳ�Ƶ��ʶ�Ӧ�Ĵ��Եĳ��ȵķֲ�
    #lennums = [[lenlist.count(i) for i in range(1,max(lenlist)+1)] 
    #           for lenlist in [[len(words2tagids[i]) for i in words2tagids]]][0]
    #lennums = [3760, 389, 32, 1]
    return words2tagids
def getpostfeatureE(weights,corps,featureTS,words2tagids):
    K = np.shape(featureTS)[0] #������
    postfeatureE = np.zeros(K) #�����ĺ�������ֵ
    N = np.shape(corps)[0]
    for corpidx in range(N):
        corp = corps[corpidx][0][1:-1]

        lencorp = np.size(corp) #���ϳ��ȣ��������еĵ�����
        Mlist = {}
        Mlist['mat'] = ['']*(lencorp+1)
        Mlist['dim'] = [words2tagids[corp[i]] for i in range(lencorp)]
        Mlist['len'] = [np.size(words2tagids[corp[i]]) for i in range(lencorp)]
        for i in range(lencorp+1):
            if i == 0:#��һ������ֻ��״̬������������
                d = Mlist['len'][0]
                Mlist['mat'][i] = np.zeros((1,d))
                for j in range(d):
                    Mlist['mat'][i][0,j] = weights[featureTS.index(('S', corp[0], words2tagids[corp[0]][j]))]        
                continue
            if i == lencorp:#���һ������Ԫ��Ϊ0������������
                Mlist['mat'][i] = np.zeros((Mlist['len'][-1],1))
                continue
            #�ȷǵ�һ��������ǵڶ�������ÿ��Ԫ��Ҫ����״̬������ת������
            Mlist['mat'][i] = np.zeros((Mlist['len'][i-1],Mlist['len'][i]))
            for d1 in range(Mlist['len'][i-1]):
                for d2 in range(Mlist['len'][i]):
                    id1 = words2tagids[corp[i-1]][d1]
                    id2 = words2tagids[corp[i]][d2]
                    try:
                        Sweight = weights[featureTS.index(('S', corp[i], id2))] 
                    except:
                        Sweight = 0
                    try:
                        Tweight = weights[featureTS.index(('T', id1, id2))]
                    except:
                        Tweight = 0
                    Mlist['mat'][i][d1,d2] = Sweight + Tweight 

        #return  Mlist,corps[0]
        #return 0
        logspace=Logspace();
        z = np.array([[0]])
        for i in range(lencorp+1):
            z = logspace.elnmatprod(z,Mlist['mat'][i])

        Alphalist = ['']*(lencorp+2)
        Betalist = ['']*(lencorp+2)
        Alphalist[0] = np.zeros((1,1))  # ��һ��ǰ��������1*1�ľ���
        Betalist[-1] = np.zeros((Mlist['len'][-1],1))
        #Alphalist���Ԫ���ǵ��о���Betalist���Ԫ���ǵ��о���
        for i in range(1,lencorp+2): 
            #print i,np.shape(Alphalist[i-1]),np.shape(Mlist['mat'][i-1])
            Alphalist[i] = logspace.elnmatprod(Alphalist[i-1],Mlist['mat'][i-1])
        for i in range(lencorp,-1,-1):
            Betalist[i] = logspace.elnmatprod(Mlist['mat'][i],Betalist[i+1])


        for i in range(1,lencorp+1):
            d1,d2 = np.shape(Mlist['mat'][i-1])
            #print d1,d2,Mlist['dim'][i-2],Mlist['dim'][i-1] # 3,2,34
            #print '================'
            for di in range(d1):
                for dj in range(d2):
                    # i=1ʱ��û��ת��������i=lencorp+1ʱ��ת��������״̬������û�� 
                    plocal = logspace.eexp(logspace.elnproduct(logspace.elnproduct(logspace.elnproduct(Alphalist[i-1][0,di],
                                                                 Mlist['mat'][i-1][di,dj]),Betalist[i][dj,0]),-z[0,0]))
                    if i == 1:#ֻ��״̬����
                        try:
                            Sidex =  featureTS.index(('S', corp[i-1], Mlist['dim'][i-1][dj]))
                            postfeatureE[Sidex] += plocal
                        except:
                            pass
                    else:
                        try:
                            Sidex =  featureTS.index(('S', corp[i-1], Mlist['dim'][i-1][dj]))
                            postfeatureE[Sidex] += plocal
                        except:
                            pass
                        try: 
                            Tidex = featureTS.index(('T', Mlist['dim'][i-2][di], Mlist['dim'][i-1][dj]))
                            postfeatureE[Tidex] += plocal
                        except:#�����ת������bucunza�����ڣ�ֱ�Ӻ���
                            pass

            #aM = logspace.elnmatprod(Alphalist[i-1],Mlist['mat'][i-1])
            #aMb = logspace.elnmatprod(aM,Betalist[i])
            #print promat
            #backuppromat.append(promat)
    postfeatureE /= N
    return postfeatureE
def getliknegvalue(weights,corps,featureTS,words2tagids):
    #Ŀ�꺯���ǶԶ�����Ȼ����ȡ������Ҫʹ����С��
    K = np.shape(featureTS)[0] #������
    N = np.shape(corps)[0]

    liknegvalue = 0

    for corpidx in range(N):
        corp = corps[corpidx][0][1:-1]
        tag = corps[corpidx][1][1:-1]

        lencorp = np.size(corp) #���ϳ��ȣ��������еĵ�����
        Mlist = {}
        Mlist['mat'] = ['']*(lencorp+1)
        Mlist['dim'] = [words2tagids[corp[i]] for i in range(lencorp)]
        Mlist['len'] = [np.size(words2tagids[corp[i]]) for i in range(lencorp)]
        for i in range(lencorp+1):
            if i == 0:#��һ������ֻ��״̬������������
                d = Mlist['len'][0]
                Mlist['mat'][i] = np.zeros((1,d))
                for j in range(d):
                    Mlist['mat'][i][0,j] = weights[featureTS.index(('S', corp[0], words2tagids[corp[0]][j]))]        
                continue
            if i == lencorp:#���һ������Ԫ��Ϊ0������������
                Mlist['mat'][i] = np.zeros((Mlist['len'][-1],1))
                continue
            #�ȷǵ�һ��������ǵڶ�������ÿ��Ԫ��Ҫ����״̬������ת������
            Mlist['mat'][i] = np.zeros((Mlist['len'][i-1],Mlist['len'][i]))
            for d1 in range(Mlist['len'][i-1]):
                for d2 in range(Mlist['len'][i]):
                    id1 = words2tagids[corp[i-1]][d1]
                    id2 = words2tagids[corp[i]][d2]
                    try:
                        Sweight = weights[featureTS.index(('S', corp[i], id2))] 
                    except:
                        Sweight = 0
                    try:
                        Tweight = weights[featureTS.index(('T', id1, id2))]
                    except:
                        Tweight = 0
                    Mlist['mat'][i][d1,d2] = Sweight + Tweight 

        numerator = 0
        logspace=Logspace()
        denominator= np.array([[0]])
        for i in range(lencorp+1):
            denominator = logspace.elnmatprod(denominator,Mlist['mat'][i])  
            if i == 0:
                numerator = logspace.elnproduct(numerator,Mlist['mat'][i][0,Mlist['dim'][i].index(tag[i])])
            elif i < lencorp:
                numerator = logspace.elnproduct(numerator,Mlist['mat'][i][Mlist['dim'][i-1].index(tag[i-1]),Mlist['dim'][i].index(tag[i])])

        liknegvalue += (denominator - numerator)/N
    return liknegvalue[0,0]
def getgradients(priorfeatureE,weights,corps,featureTS,words2tagids):
    postfeatureE = getpostfeatureE(weights,corps,featureTS,words2tagids)
    return postfeatureE - priorfeatureE

#L-BFGS����������ֵ�Ż�
def twoloop(s, y, rho,gk):
    # ��lbfgs��������
    n = len(s) #�������еĳ���
    if np.shape(s)[0] >= 1:
        #h0�Ǳ��������Ǿ���
        h0 = 1.0*np.dot(s[-1],y[-1])/np.dot(y[-1],y[-1])
    else:
        h0 = 1
    a = np.empty((n,))
    q = gk.copy() 
    for i in range(n - 1, -1, -1): 
        a[i] = rho[i] * np.dot(s[i], q)
        q -= a[i] * y[i]
    z = h0*q
    for i in range(n):
        b = rho[i] * np.dot(y[i], z)
        z += s[i] * (a[i] - b)

    return z   

def lbfgs(fun = getliknegvalue,gfun = getgradients,x0=0,corps = 0,
          featureTS = 0,words2tagids = 0,
          priorfeatureE = 0,m=10,maxk = 20):
    # fun��gfun�ֱ���Ŀ�꺯������һ�׵���,x0�ǳ�ֵ,mΪ��������еĴ�С
    rou = 0.55
    sigma = 0.4
    epsilon = 1e-5
    k = 0
    n = np.shape(x0)[0] #�Ա�����ά��

    s, y, rho = [], [], []

    while k < maxk :

        gk = gfun(priorfeatureE,x0,corps,featureTS,words2tagids)
        if np.linalg.norm(gk) < epsilon:
            break

        dk = -1.0*twoloop(s, y, rho,gk)

        m0=0;
        mk=0
        funcvalue = fun(x0,corps,featureTS,words2tagids)
        while m0 < 20: # ��Armijo�����󲽳�
            if fun(x0+rou**m0*dk,corps,featureTS,words2tagids) < funcvalue+sigma*rou**m0*np.dot(gk,dk): 
                mk = m0
                break
            m0 += 1


        x = x0 + rou**mk*dk
        sk = x - x0
        yk = gfun(priorfeatureE,x,corps,featureTS,words2tagids) - gk   

        if np.dot(sk,yk) > 0: #�����µ�����
            rho.append(1.0/np.dot(sk,yk))
            s.append(sk)
            y.append(yk)
        if np.shape(rho)[0] > m: #�����������
            rho.pop(0)
            s.pop(0)
            y.pop(0)

        k += 1
        x0 = x
        #print("iteration times��%d, function value��%f"%(k,funcvalue))
    return x0, fun(x0,corps,featureTS,words2tagids)#,k#�ֱ������ŵ����꣬����ֵ����������

