#coding=gbk
import matplotlib.pyplot  as plt
from math import log
import operator
class trees(object):
    """description of class"""
    def __init__(self):
        pass
    # 
    def calcShannonEnt(self,Dataset):
        #计算香农熵
        numEntries = len(Dataset)
        count_label = {}
        #统计标签相同的记录数，用字典count_label记录
        #statistics the numbers of records with same label in last column
        for featVect in Dataset:
            cur_label = featVect[-1]
            if cur_label not in count_label.keys():
               count_label[cur_label] = 0
            count_label[cur_label]+=1
        ShannonEnt = 0.0
        for key in count_label:
            prob = float(count_label[key]) / numEntries
            ShannonEnt-=prob * log(prob,2)  # 数据标签越一致，香农熵越小，最好为
        print('The entropy of the DataSet have been calculated:',ShannonEnt)
        return ShannonEnt
    # 
    def creatDataSet(self):
        retnDataSet = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'maybe']]
        labels = ['no surfacing','flippers']
        return retnDataSet,labels
    # 
    def splitDataSet(self,DataSet,axis,value):
       #Extract those who have same value in column axis to form sub-DataSet
       print('=============$splitDataSet========================')
       print('These records, which have "',value,'" in column "',axis,'", will be extracted!')
       retDataSet = []  
       for featVect in DataSet:
             if  featVect[axis] == value:
                  reduced = featVect[:axis]
                  reduced.extend(featVect[axis + 1:]) 
                  retDataSet.append(reduced)
       print('=============$End splitDataSet========================')
       return retDataSet  
    # 
    def chooseBestFeature2Split(self,dataSet):
           # choose the best Feature from all column of dataSet,which 
           # can split dataset with minmal infomation gain 
              print('-----------$chooseBestFeature2Split-----------')
              numEntires = len(dataSet)
              numFeat = len(dataSet[0]) - 1
              bestGain = 0.0;bestFeature=-1
              baseEnt = self.calcShannonEnt(dataSet)
              for i in range(numFeat):
                     alLabel = [x[i] for x in dataSet]
                     uniqueLabel = set(alLabel)    
                     newEntropy = 0.0 
                     for ul in uniqueLabel:  #为每一个取值都计算信息增益 
                           subSet = self.splitDataSet(dataSet,i,ul)
                           ratio = len(subSet) / numEntires
                           newEntropy+=ratio * self.calcShannonEnt(subSet)
                     infoGain = baseEnt - newEntropy
                     if   infoGain > bestGain:
                            bestGain = infoGain
                            bestFeature = i
              print('-----------$End chooseBestFeature2Split-----------')
              return bestFeature
    def majorityCnt(self,classList):
      classCount = {}
      print("*****************$majorityCnt****************")
      print(classList)
      for   vote  in   classList:
              if vote not in  classCount.keys():
                    classCount[vote] = 0
                    print(vote)
                    print(classCount)
              classCount[vote]+=1
      sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
      print(sortedClassCount,'\t type:',type(sortedClassCount))  
      #as a whole, this variable is  a list , the elements in it is tuple
      print("*****************$ End majorityCnt****************")
      return sortedClassCount[0][0]
    # 
    def createTree(self,dataSet,labels):     
      print("-------------------------reporter----------------------------------------")
      print("dataSet:",dataSet,"\nlabels:",labels)
      backupLabel=labels.copy() #if  assignment in here, the content of variable backupLabel \
                                                   #will be changed while varable labels be changed 
      classList = [example[-1] for example in dataSet]
      if classList.count(classList[0]) == len(classList):  #the finishing condition of recurion either the same last column 
            print("the last(result) column has been distinguished competely!") 
            return classList[0]
      if    len(dataSet[0]) == 1:  #    or  only one column remained
            return self.majorityCnt(classList)
      bestFeature = self.chooseBestFeature2Split(dataSet)  # 
    
      print("The best feature index has been choosed as ",(bestFeature),str("("+backupLabel[bestFeature]+")"))
      bestFeatLab = labels[bestFeature]
      myTree = {bestFeatLab:{}}
      print("The current tree is :\n\t",str(myTree))
      del(labels[bestFeature]) # 将已经使用的特征删除
      featValues = [example[bestFeature] for example in dataSet]    #value of column
      uniqueValues = set(featValues)
      print(" The feature values of ",bestFeature,"(",backupLabel[bestFeature], ")"," is : ",str(uniqueValues))
      print("-----------------------------------------------------------------")    
      for unqVal in   uniqueValues:  # Constuct the brunch for the current feature
                subLabels = labels[:]
                myTree[bestFeatLab][unqVal] = self.createTree(self.splitDataSet(\
                      dataSet,bestFeature,unqVal),subLabels)      #the indice  for a dict object ,on 1-d index as str-key
      print("##############################################")
      print("The subtree of current feature ",bestFeature,backupLabel[bestFeature],"has been built:\n\t ")
      print(myTree)
      print("##############################################")
      return myTree



    def renderData(self):
         f = open("./Ch03/lenses.txt")
         allines = f.readlines()
         lenses=[inst.strip().split('\t') for inst in allines]
         labels=['age','prescript','astigmatic','tearRate']
         return  lenses,labels
    def classify(self,inputTree,featLabels,testVect) :
                    firstStr=inputTree.keys()[0]
                    secondDict=inputTree[firstStr]
                    featIndex=featLabels.index[firstStr]
                    for key in secondDict.keys()  :
                           if testVect[featIndex]==key:
                                  if type(secondDict[key]).__name__=='dict':
                                          classify(secondDict[key],featLabels,testVect)
                                  else:
                                        classLabel=secondDict[key]
                    return classLabel                        
   ##--------------------------------------------------------------------------------------
   ##  
   ##--------------------------------------------------------------------------------------
   # def getPlotProperties(self):
    decisionNode = dict(boxstyle="sawtooth",function="0.8")
    leafNode = dict(boxstyle="round4",function="0.8")
    arrow_args = dict(arrowstyle="<-")
          #return  decisionNode, leafNode,  arrow_args
    def plotNode(self,nodeTxt,centerPt,parentPt,nodeType):
          arrow_args = dict(arrowstyle="<-")
          self.axl.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',\
               va="center",ha="center",xytext=centerPt,#表示垂直或水平对齐的方式
               textcoords='axes fraction',bbox=nodeType,\
               arrowprops=arrow_args)# boxstyle=nodeType,va="center", ha="center",
    def getNumLeafs(self,myTree):
            numLeafs = 0
            t=myTree
            firstStr = list(myTree.keys() )[0]
            print("-------------------------------------------")
            print(firstStr)
            secondDict = myTree[firstStr]
            for key in secondDict.keys():
                 if type(secondDict[key]).__name__ == 'dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
                       numLeafs += self.getNumLeafs(secondDict[key])
                 else:   numLeafs +=1
            return numLeafs
      # 
    def getTreeDepth(self,myTree):
                 maxDepth = 0
                 firstStr = list(myTree.keys())[0]
                 secondDict = myTree[firstStr]
                 for key in secondDict.keys():
                            if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes
                                     thisDepth =1+self.getTreeDepth(secondDict[key])
                            else:thisDepth = 1
                 if thisDepth > maxDepth: maxDepth = thisDepth
                 return maxDepth
    def retrieveTree(self,i):
           ListofTree=[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                             {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}} ]
           return ListofTree[i]
     #
    def plotMidText(self,cntrPt, parentPt, txtString):
            xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
            yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
            self.axl.text(xMid, yMid, txtString)#va="center", ha="center",,rotation=30
    def plotTree(self,myTree, parentPt, nodeTxt):   #if the first key tells you what feat was split on
          
             decisionNode = dict(boxstyle="sawtooth",fc="0.8")
             leafNode = dict(boxstyle="round",fc="0.8")  #define the type of plotted node 
             arrow_args = dict(arrowstyle="<-")
             numLeafs = self.getNumLeafs(myTree)         #this determines the x width of this tree
             depth = self.getTreeDepth(myTree)
             firstStr = list(myTree.keys())[0]     #the text label for this node should be this
             cntrPt = (self.xOff + (1.0 + float(numLeafs))/2.0/self.totalW, self.yOff)
             self.plotMidText(cntrPt, parentPt, nodeTxt)
             self.plotNode(firstStr, cntrPt, parentPt, decisionNode)
             secondDict = myTree[firstStr]
             self.yOff = self.yOff - 1.0/self.totalD
             for key in secondDict.keys():
                  if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
                          self.plotTree(secondDict[key],cntrPt,str(key))        #recursion
                  else:   #it's a leaf node print the leaf node
                       self.xOff = self.xOff + 1.0/self.totalW
                       self.plotNode(secondDict[key], (self.xOff, self.yOff), cntrPt, leafNode)
                       self.plotMidText((self.xOff, self.yOff), cntrPt, str(key))
             self.yOff = self.yOff + 1.0/self.totalD
               #if you do get a dictonary you know it's a tree, and the first element will be another dict
    def createPlot(self,inTree):
    
          fig = plt.figure(1, facecolor='white')
          fig.clf()
          axprops = dict(xticks=[], yticks=[])
          self.axl= plt.subplot(111, frameon=False, **axprops)    #no ticks
           #createPlot.ax1 = plt.subplot(111, frameon=False)    #ticks for demo puropses 
          self.totalW = float(self.getNumLeafs(inTree))
          self.totalD = float(self.getTreeDepth(inTree))
          self.xOff = -0.5/self.totalW; 
          self.yOff = 1.0;
          self.plotTree(inTree, (0.5,0.9), '')
          plt.show()







               