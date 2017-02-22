from votesmart import votesmart
class Apriori:
   def __init__(self):
       pass
   def loadDataSet(self):
       return [[1,3,4],[2,3,5],[1,2,3,5],[2,5]]
   def CreateC1(self,dataSet):
     c1=[]
     for transcation in dataSet:
         for item in transcation:
              if [item] not in c1:
                   c1.append([item])
     c1.sort()
     return list(map(frozenset,c1))
   def scanD(self,D,ck,minSupport):
     ssCnt={}
     numItems=0
     for tid in D:
         numItems+=1
         for candicate in ck:
               if candicate.issubset(tid):
                    if  candicate not in ssCnt.keys():
                        ssCnt[candicate]=1
                    else: ssCnt[candicate]+=1

     retList=[]
     supportData={}
     for key in ssCnt:
          support=ssCnt[key]/numItems
          if support>=minSupport:
               retList.insert(0,key)
          supportData[key]=support
     return retList,supportData
      #Li=retList is that  include i elements like [frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})]
      #supportData is support numbers of each element in ck  
   def aprioriGen(self,Lk,k):
       retList=[]
       lenLk=len(Lk)
       for i in range(lenLk):
             for j in range(i+1,lenLk):
                 L1=list(Lk[i])[:k-2]; L2=list(Lk[j])[:k-2];
                 L1.sort();L2.sort()
                 if L1==L2:
                      retList.append(Lk[i]|Lk[j])
       return retList
   def apriori(self,dataSet,minSupport=0.5):
        C1=self.CreateC1(dataSet)# 生成元素集合，frozenset 相当于const
        # C1 like "[frozenset({1}), frozenset({2}), frozenset({3}), frozenset({4}), frozenset({5})]"
        D=list(map(set,dataSet)) 
        #D like [[1, 3, 4], [2, 3, 5], [1, 2, 3, 5], [2, 5]]-->[{1, 3, 4}, {2, 3, 5}, {1, 2, 3, 5}, {2, 5}]
        #将每个list 变成集合，相当于去重复

        L1,supportData=self.scanD(D,C1,minSupport)
        #在 D 中寻找C1 中支持度不小于minSupport的项集，supportData记录C1 中每个项集的支持度
        #Li is that  include i elements like [frozenset({1}), frozenset({3}), frozenset({2}), frozenset({5})]

        L=[L1]
        k=2
        while(len(L[k-2])>0):
             Ck=self.aprioriGen(L[k-2],k)
             Lk,supk=self.scanD(D,Ck,minSupport)
             supportData.update(supk)
             L.append(Lk)
             k+=1
        return L,supportData
   def generateRules(self,L,supportData,minConf=0.7):
       bigRuleList=[]
       for i in range(1,len(L)):
            for freqSet in L[i]:#L[i] denotes frequency set that include i+1 elements  
                H1=[frozenset([item]) for item in freqSet]
                if (i>1):
                     self.rulesFromConseq(freqSet,H1,supportData,bigRuleList,minConf)
                else:
                    self.calcConf(freqSet,H1,supportData,bigRuleList,minConf)
       return bigRuleList             
 
   def rulesFromConseq(self,freqSet,H,supportData,brl,minConf=0.7):
         # freqSet=[2,3,5] ,H=[[2],[3],[5]]
         m=len(H[0])
         if(len(freqSet)>(m+1)):
              Hmp1=self.aprioriGen(H,m+1)
              Hmp1=self.calcConf(freqSet,Hmp1,supportData,brl,minConf)
              if(len(Hmp1)>1):
                   self.rulesFromConseq(freqSet,Hmp1,supportData,brl,minConf)


   def calcConf(self,freqSet,H,supportData,brl,minConf=0.7):
          prunedH=[]
          for conseq in H:
               conf=supportData[freqSet]/supportData[freqSet-conseq]
               if conf>= minConf:
                    print(freqSet-conseq,'-->',conseq,'conf:',conf)
                    brl.append((freqSet-conseq,conseq,conf))
                    prunedH.append(conseq)
          return prunedH

   def debug(self):
       dataSet=self.loadDataSet()
       #c1=self.CreateC1(dataSet)
       #D=list(map(set,dataSet))
       #retList,supportData=self.scanD(D,c1,0.5) 
       L,supportData=self.apriori(dataSet)
       LL=list(L)
       print("======================================================")
       for i in range(len(LL)):
             print("\t包含 ",i+1,"个元素的频繁项集：",LL[i])
          
       for key in supportData.keys():
             print("\t项集",list(key),"出现的概率：",supportData[key])
       rules=self.generateRules(L,supportData,minConf=0.7)
      
       mushDataSet=[ line.split() for line in open('./Ch11/mushroom.dat').readlines()]
   
       fs,supportData=self.apriori(mushDataSet)

       a=0


          



        



