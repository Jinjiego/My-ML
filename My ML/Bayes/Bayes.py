import   numpy as npy
from math import log
import re

class Bayes(object):
    """description of class"""
    def loadDataSet(self):
         postingList=\
                 [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], 
                 ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], 
                 ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],             
                 ['stop', 'posting', 'stupid', 'worthless', 'garbage'], 
                 ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], 
                 ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
         classVect=[0,1,0,1,0,1]    #1 stands for  insulting comment and 0 normal comment
         return postingList,classVect
      #create an undump list set for all words in a passage 
    def createVocabraryList(self,dataSet):
           VocaSet=set([])
           for document in dataSet: # document is a line in dataSet if dataSet is a 2 dimension list
                  VocaSet=VocaSet|set(document)  #combination operation  AUB
           return list(VocaSet)
    def setOfWord2Vex(self,vocaList,doc):
          #vocaList  :vocabrary list
          #doc :a specified document
          #function: every word in vocaList whether or not in   doc
          #return a Vector with same size to vocaList
          retVect=[0]*len(vocaList)
          for word in doc:
                if word in vocaList:
                      retVect[vocaList.index(word)]=1
                else:
                      print("the word %s is not in vocaList",word)  
          return retVect
    def renderTrain(self):
            trainMat=[]
            postingList,classList=self.loadDataSet()
            vocabList=self.createVocabraryList(postingList)
            for rowDoc in postingList:
                   trainMat.append(self.setOfWord2Vex(vocabList,rowDoc) )
            return trainMat
    def trainNB0(self,trainMat,classList):
           numRows=len(trainMat)
           numCols=len(trainMat[0])
           pAbusive=sum(classList)/float(numRows)
           p0Num=npy.ones(numCols); p1Num=p0Num.copy()
           p0Denom=2.0 ; p1Denom=2.0
           for i  in range(numRows):
                  if  classList[i]==1 : #judge what is type of this document?
                         p1Num+=trainMat[i]
                         p1Denom+=sum(trainMat[i])
                  else:
                        p0Num+=trainMat[i]
                        p0Denom+=sum(trainMat[i])
           
           print("p0Num:\n",p0Num)
           print("p0Denom:",p0Denom)  
           print("p1Num:\n",p1Num)
           print("p1Denom:",p1Denom)
           p1Vect=npy.log(p1Num/p1Denom)
           p0Vect=npy.log(p0Num/p0Denom)
           return p0Vect,p1Vect,pAbusive
    def classifyNB0(self,vectDoc,p0v,p1v,p_c1):
            print(vectDoc,p0v,p1v,p_c1)
            p1=sum(p1v*vectDoc)+log(p_c1)
            p0=sum(p0v*vectDoc)+log(1-p_c1)
            print("the given document  p1=",p1,",p0=",p0)
            if p1>p0 :
                  return 1
            else:
                  return 0
    def bagOfWords2VectMN(self,vocabList,Doc):
          bagVect=[0]*len(vocabList)
          for word in Doc:
                if word in vocabList:
                        bagVect[vocabList.index(word)]+=1
          return bagVect
    def textParser(self,bigString):
            regEx=re.compile('\\W*')
            bigStrListTemp= regEx.split(bigString)
            bigStrList=[tok for tok in bigStrListTemp if len(tok)>0]
    