

class FP_growth():
    def __init__(self): 
        pass
         
    def createTree(self,dataSet,minSup=1,log=True):
          '''dataSet={frozenset({'z'}): 1,  frozenset({'z', 'r', 'p', 'h', 'j'}): 1,
                      frozenset({'s', 't', 'x', 'v', 'u', 'y', 'w', 'z'}): 1, 
                      ...
                      frozenset({'o', 'r', 'n', 'x', 's'}): 1}'''
          HeaderTable ={}
          HeaderTable_fliter={}
          ##==========1  scanf dataset firstly for finding frequency items======
          for trans in dataSet:# trans=frozenset({'o', 'r', 'n', 'x', 's'})
               for item in trans:
                    HeaderTable[item]=HeaderTable.get(item,0)+dataSet[trans]
          ##HeaderTable={'e': 1, 'h': 1, 'j': 1, 'm': 1, 'n': 1, 'o': 1, 'p': 2, 'q': 2, 'r': 3, 's': 3, 't': 3, 'u': 1, 'v': 1, 'w': 1, ...}
          for k in HeaderTable.keys():
               if HeaderTable[k]<minSup:
                    HeaderTable_fliter[k]=HeaderTable[k]
          #HeaderTable_fliter={'e': 1, 'h': 1, 'j': 1, 'm': 1, 'n': 1, 'o': 1, 'p': 2, 'q': 2, 'u': 1, 'v': 1, 'w': 1}
          for k in HeaderTable_fliter:
               del(HeaderTable[k])
         
          freqItemSet=set(HeaderTable.keys())
          if(len( freqItemSet)==0 ):  return None,None
          for k in HeaderTable.keys():
                HeaderTable[k]=[HeaderTable[k] ,None]  #table header
          retTree=treeNode('None Set',1,None)
          # scanf data secondly for 
          for transSet,count in dataSet.items():#transSet= frozenset({'s', 't', 'x', 'v', 'u', 'y', 'w', 'z'})
               localD={}
               for item in transSet:
                   if item in freqItemSet:
                         localD[item]=HeaderTable[item][0]
               if len(localD)>0:
                    tmp=sorted(localD.items(),key=lambda p:p[1],reverse=True)
                    #difference order among items with same support maybe influence shape of fp-tree
                    orderedItems=[v[0] for v in tmp]
                    if log:
                       print("从",transSet,"中获得频繁集",tmp,"加入树中")
                    self.updateTree(orderedItems,retTree,HeaderTable,count)
          return  retTree,HeaderTable
    def updateTree(self,items,inTree,headerTable,count):
        # ietms -['z', 'r'],
        # headerTable={'r': [3, None], 's': [3, None], ...}

        if items[0] in inTree.children :# children is a dictionary
             inTree.children[items[0]].inc(count) 
        else:
             inTree.children[items[0]]=treeNode(items[0],count,inTree)# children also index it's child node with items
             if headerTable[items[0]][1]==None:# link with header
                  headerTable[items[0]][1]=inTree.children[items[0]]
             else: 
                  self.updateHeader(headerTable[items[0]][1],inTree.children[items[0]])
        if len(items)>1:
             self.updateTree(items[1::],inTree.children[items[0]],headerTable,count)

    def updateHeader(self,node2Test,targetNode ):
         while(node2Test.nodeLink!=None):
              node2Test=node2Test.nodeLink
         node2Test.nodeLink=targetNode
    def loadDataSet(self):
        simpDat = [['r', 'z', 'h', 'j', 'p'],
               ['z', 'y', 'x', 'w', 'v', 'u', 't', 's'],
               ['z'],
               ['r', 'x', 'n', 'o', 's'],
               ['y', 'r', 'x', 'z', 'q', 't', 'p'],
               ['y', 'z', 'x', 'e', 'q', 's', 't', 'm']]
        return simpDat
    def ascendTree(self,leafNode,prefixPath):
        if leafNode.parent!=None:
             prefixPath.append(leafNode.name)
             self.ascendTree(leafNode.parent,prefixPath)
    def findPrefixPath(self,basePat,treeNode):
         condPats={}
         while treeNode!=None:
             prefixPath=[]
             self.ascendTree(treeNode,prefixPath)
             if len(prefixPath)>1:
                  condPats[ frozenset(prefixPath[1:] ) ]=treeNode.count
             treeNode=treeNode.nodeLink
         return condPats

    def createInitSet(self,dataSet):
        retDict={}
        for trans in dataSet:
             retDict[frozenset(trans)]=1
        return retDict
    def mineTree(self,inTree,hearderTable,minSup,prefix,freqItemList,log=True):
        bigL=[v[0] for v in sorted(hearderTable.items(),key=lambda p:p[0])]
        for basePat in bigL:
             newFreqSet=prefix.copy()
             newFreqSet.add(basePat)
             freqItemList.append(newFreqSet)
             condPattBases=self.findPrefixPath(basePat,hearderTable[basePat][1])
             myCondTree,myHead=self.createTree(condPattBases,minSup,log)
             if myHead!=None:
                 if log:
                      print('conditional tree for:',newFreqSet)
                      myCondTree.disp()
                 self.mineTree(myCondTree,myHead,minSup,newFreqSet,freqItemList)

    def debug(self):
        dataSet=self.loadDataSet()
        retDict=self.createInitSet(dataSet)
        myFpTree,myHeaderTable=self.createTree(retDict,3)
        print("=============prefix path for each frequency items============================")
        for k in myHeaderTable:
             k_prefix=self.findPrefixPath(k,myHeaderTable[k][1])
             print(k,"的前缀路径：",k_prefix)
  
        print("=============mine tree================================")
        freqItemList=[]
        self.mineTree(myFpTree,myHeaderTable,3,set([]),freqItemList)

        print("========================================================")
        print("reading data...")
        parsedDat=[line.split() for line in open('./Ch12/kosarak.dat').readlines()]
        print("initilizing data and create fp-tree...")
        InitSet=self.createInitSet(parsedDat)
        myKosTree,myKosHeaderTab=self.createTree(InitSet,100000,False)
        myFreqList=[]
        print("mining fp-tree...")
        self.mineTree( myKosTree,myKosHeaderTab,100000,set([]),myFreqList,False)
        len(myFreqList)
        print(len(myFreqList),":",myFreqList)
        a=0 



          
class treeNode(object):
      def __init__(self,nameValue,numOccur,parentNode):
           self.name=nameValue
           self.count=numOccur
           self.nodeLink=None
           self.parent=parentNode
           self.children={} 
      def inc(self,numOccur):
           self.count+=numOccur
      def disp(self,ind=1):
          print(' '*ind,self.name,'',self.count)
          for child in self.children.values():
               child.disp(ind+1)

