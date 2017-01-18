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
        numEntries = len(Dataset)
        count_label = {}
        for featVect in Dataset:
            cur_label = featVect[-1]
            if cur_label not in count_label.keys():
               count_label[cur_label] = 0
            count_label[cur_label]+=1
        ShannonEnt = 0.0
        for key in count_label:
            prob = float(count_label[key]) / numEntries
            ShannonEnt-=prob * log(prob,2)
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
       retDataSet = []  
       for featVect in DataSet:
             if  featVect[axis] == value:
                  reduced = featVect[:axis]
                  reduced.extend(featVect[axis + 1:]) 
                  retDataSet.append(reduced)
       return retDataSet  
      # 
    def chooseBestFeature2Split(self,dataSet):
              numEntires = len(dataSet)
              numFeat = len(dataSet[0]) - 1
              bestGain = 0.0
              baseEnt = self.calcShannonEnt(dataSet)
              for i in range(numFeat):
                     alLabel = [x[i] for x in dataSet]
                     uniqueLabel = set(alLabel)    
                     newEntropy = 0.0 
                     for ul in uniqueLabel:
                           subSet = self.splitDataSet(dataSet,i,ul)
                           ratio = len(subSet) / numEntires
                           newEntropy+=ratio * self.calcShannonEnt(subSet)
              infoGain = baseEnt - newEntropy
              if   infoGain > bestGain:
                   bestGain = infoGain
                   bestFeature = i
              return bestFeature
    def majorityCnt(self,classList):
      classCount = {}
      print("---------------majorityCnt----------------------")
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
    
      print("the best feature index has been choosed as ",(bestFeature),str("("+backupLabel[bestFeature]+")"))
      bestFeatLab = labels[bestFeature]
      myTree = {bestFeatLab:{}}
      print("the current tree is :\n\t",str(myTree))
      del(labels[bestFeature])
      featValues = [example[bestFeature] for example in dataSet]    #value of column
      uniqueValues = set(featValues)
      print(uniqueValues)
      print(" the feature values of ",bestFeature,"(",backupLabel[bestFeature], ")"," is : ",str(uniqueValues))
      print("-----------------------------------------------------------------")    
      for unqVal in   uniqueValues:
                subLabels = labels[:]
                myTree[bestFeatLab][unqVal] = self.createTree(self.splitDataSet(\
                      dataSet,bestFeature,unqVal),subLabels)      #the indice  for a dict object ,on 1-d index as str-key
      print("##############################################")
      print("the subtree of current feature ",bestFeature,backupLabel[bestFeature],"has been built:\n\t ")
      print(myTree)
      print("##############################################")
      return myTree
    def renderData(self):
         f = open("lenses.txt")
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
          global axl
          arrow_args = dict(arrowstyle="<-")
          axl.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
               xytext=centerPt, textcoords='axes fraction',
               va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
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
                                     thisDepth = 1 + self.getTreeDepth(secondDict[key])
                            else:   thisDepth = 1
                 if thisDepth > maxDepth: maxDepth = thisDepth
                 return maxDepth
    def retrieveTree(self,i):
           ListofTree=[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', 1: 'yes'}}}},
                             {'no surfacing': {0: 'no', 1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}} ]
           return ListofTree[i]
     #
    def plotMidText(self,cntrPt, parentPt, txtString):
            global axl
            xMid = (parentPt[0]-cntrPt[0])/2.0 + cntrPt[0]
            yMid = (parentPt[1]-cntrPt[1])/2.0 + cntrPt[1]
            axl.text(xMid, yMid, txtString, va="center", ha="center", rotation=30)
    def plotTree(self,myTree, parentPt, nodeTxt):   #if the first key tells you what feat was split on
             global xOff
             global yOff
             global  totalW
             global  totalD
             decisionNode = dict(boxstyle="sawtooth",function="0.8")
             leafNode = dict(boxstyle="round4",function="0.8")
             arrow_args = dict(arrowstyle="<-")
             numLeafs = self.getNumLeafs(myTree)         #this determines the x width of this tree
             depth = self.getTreeDepth(myTree)
             firstStr = list(myTree.keys())[0]     #the text label for this node should be this
             cntrPt = (xOff + (1.0 + float(numLeafs))/2.0/totalW, yOff)
             self.plotMidText(cntrPt, parentPt, nodeTxt)
             self.plotNode(firstStr, cntrPt, parentPt, decisionNode)
             secondDict = myTree[firstStr]
             yOff = yOff - 1.0/totalD
             for key in secondDict.keys():
                  if type(secondDict[key]).__name__=='dict':#test to see if the nodes are dictonaires, if not they are leaf nodes   
                          self.plotTree(secondDict[key],cntrPt,str(key))        #recursion
                  else:   #it's a leaf node print the leaf node
                       plotTree.xOff = xOff + 1.0/totalW
                       plotNode(secondDict[key], (xOff, yOff), cntrPt, leafNode)
                       plotMidText((xOff, yOff), cntrPt, str(key))
                  plotTree.yOff = yOff + 1.0/totalD
               #if you do get a dictonary you know it's a tree, and the first element will be another dict
    def createPlot(self,inTree):
          global xOff
          global yOff
          global  totalW
          global  totalD
          global axl
          fig = plt.figure(1, facecolor='white')
          fig.clf()
          axprops = dict(xticks=[], yticks=[])
          axl= plt.subplot(111, frameon=False, **axprops)    #no ticks
           #createPlot.ax1 = plt.subplot(111, frameon=False)    #ticks for demo puropses 
          totalW = float(self.getNumLeafs(inTree))
          totalD = float(self.getTreeDepth(inTree))
          xOff = -0.5/totalW; 
          yOff = 1.0;
          self.plotTree(inTree, (0.5,1.0), '')
          plt.show()







               