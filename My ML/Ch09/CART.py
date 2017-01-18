import numpy as ny
import matplotlib.pyplot as plt
class CART(object):
    """description of class"""
    def __init__(self):
        pass
    def LoadDataSet(self,fn):
        datMat=[]
        fr=open(fn)
        for  row in fr.readlines():
            curLine=row.strip().split('\t')
            fltLine=list(map(float,curLine))
            datMat.append(fltLine)

        return datMat
    def binSplitDataSet(self,dataSet,feature,value):
          mat0=dataSet[ny.nonzero( dataSet[:,feature]>value)[0],:]
          mat1=dataSet[ny.nonzero( dataSet[:,feature]<=value)[0],:]
          return mat0,mat1 
    def  regLeaf(self,dataSet): 
          return ny.mean(dataSet[:,-1])
    def  regErr(self,dataSet):
          return ny.var( dataSet[:,-1] *ny.shape(dataSet)[0] )  
    def linearSlover(self,dataSet):
        #Input: dataSet shoule be  a matrix of numpy.mat
        # This function perform a linear regression
         m,n=ny.shape(dataSet)
         X=ny.mat(ny.ones((m,n)))
         X=dataSet[:,0:n-1]
         Y=X[:,0]; Y=dataSet[:,-1]
         xTx=X.T*X 
         if ny.linalg.det(xTx)==0.0:
               raise NameError("This matrix is singular,cannot do inverse,\
               try increasing the second value of ops")
         w=xTx.I*X.T*Y
         return w,X,Y
    def modelLeaf(self,*kw):  # *a 会把参数组成为toupe,**a 会把参数组成为dict
         dataSet=kw[-1];
         w,X,Y=self.linearSlover(dataSet)
         return w
    def modelErr(self,*kw):
         dataSet=kw[-1]
         w,X,Y=self.linearSlover(dataSet)
         yHat=X*w
         return ny.sum( ny.power(Y-yHat,2 ) ) # 计算误差
    def chooseBestSplit(self,dataSet,leafType,errType,ops):
           tolS=ops[0];tolN=ops[1]
           if len( set( dataSet[:,-1].T.tolist()[0] ) )==1:
                return None,leafType(self,dataSet )
           m,n=ny.shape(dataSet)
           S=errType(self,dataSet)
           bestS=ny.inf; bestIndex=0; bestValue=0

           for featIndex in range(n-1):
                 #===========lambda L:L[0] will return a function handle======================
                 #       which yielded value specfied by expression
                 #       following sentence only for get a list from a column in a matrix
                 #       function map used for transforming [[...],[...],...,[...]] to [............]
                 #============================================================================
                 temp=list(map(lambda L:L[0],dataSet[:,featIndex].tolist()   ))
                 #===============split the dataSet with each value in feature======
                 #===the generated spliting with minmal error will be accepted for best partition=====  
                 for splitVal in set(temp): 
                      mat0,mat1=self.binSplitDataSet( dataSet,featIndex,splitVal )
                      if ny.shape(mat0)[0]<tolN or ny.shape(mat1)[0]<tolN :
                           continue
                      newS=errType(self,mat0)+errType(self,mat1)
                      if newS<bestS: 
                           bestIndex=featIndex
                           bestValue=splitVal
                           bestS=newS
                 #=================================================================
           if (S-bestS)<tolS:
                return None,leafType(self,dataSet)
           mat0,mat1=self.binSplitDataSet( dataSet,bestIndex,bestValue)
           if (ny.shape(mat0)[0] < tolN) or ( ny.shape( mat1)[0] < tolN ):
                  return None,leafType(self,dataSet)
           return bestIndex,bestValue 
            
    def createTree(self,dataSet,leafType=regLeaf,errType=regErr,ops=(1,4)):
          feat,val=self.chooseBestSplit(dataSet,leafType,errType,ops)   
          if feat==None:
                return val
          retTree={}
          retTree['spInd']=feat
          retTree['spVal']=val

          lSet,rSet=self.binSplitDataSet(dataSet,feat,val)
          retTree['left']=self.createTree(lSet,leafType,errType,ops )
          retTree['right']=self.createTree(rSet,leafType,errType,ops)
          return retTree                     
    def Invoker(self):
        datfn="./Ch09/ex0.txt"
        dataft='./Ch09/ex2test.txt'
        datMat=self.LoadDataSet(datfn)
        data2Test=self.LoadDataSet(dataft)  
        mydatMat=ny.mat(datMat)
        data2TestMat=ny.mat(data2Test)
        tr=self.createTree(mydatMat)
      
        #self.getMean(tr)
        self.prune(tr,data2TestMat)
       
        model_tree=self.createTree(mydatMat,self.modelLeaf,self.modelErr)
        
        bikeSpeedVsIq_train=self.LoadDataSet("./Ch09/bikeSpeedVsIq_train.txt")

        plt.scatter([x[0] for x in bikeSpeedVsIq_train],[x[1] for x in bikeSpeedVsIq_train])
        plt.show()
        bikeSpeedVsIq_tree=self.createTree(ny.mat(bikeSpeedVsIq_train),)



        a=0

    def isTree(self,obj):
        return type(obj).__name__=='dict'
    def getMean(self,tree):
        if self.isTree( tree['left']): 
            tree['left']=self.getMean(tree['left'])
        if self.isTree(tree['right']):
            tree['right']=self.getMean(tree['right'])

        return ( tree['left']+tree['right'])/2.0
    def prune(self,tree,testData): 
       if ny.shape(testData)[0]==1:
            return self.getMean(tree)
       if self.isTree(tree['left']) or self.isTree(tree['right']):
              lSet,rSet=self.binSplitDataSet(testData,tree['spInd'],tree['spVal'])
       ##
       if self.isTree(tree['left']):
           tree['left']=self.prune(tree['left'],lSet)
       if self.isTree(tree['right']):
           tree['right']=self.prune(tree['right'],rSet)
       if not self.isTree(tree['left']) and not self.isTree(tree['right']):
             lSet,rSet=self.binSplitDataSet(testData,tree['spInd'],tree['spVal'])
             errorNomerge=sum(ny.power( lSet[:,-1]-tree['left'],2) )+\
                          sum(ny.power( rSet[:,-1]-tree['right'],2)) 
             treeMean=(tree['left']+tree['right'])/2.0
             errorMerge=sum( ny.power( testData[:,-1]-treeMean,2) )
             if errorMerge<errorNomerge:
                 print("merging")
                 return treeMean
             else: return tree
       else:
           return tree
    def regTreeEval(self,**kw):
         model=kw['model']
         return float(model )
    def modelTreeEval(self,*kw): 
         model=kw['model']
         InData=kw['InData']
         w=model
         m,n=ny.shape(InData)
         x=ny.mat(ny.ones( (m,n+1) ))
         x[:,1:n+1]=ny.mat(InData)
         return float(w*x)
         
    def treeForeCast(self,tree,InData,modelEval=regTreeEval):
        if not self.isTree(tree):
             return  modelEval(model=tree,inData=InData)
        if InData[tree['spInd']]>tree['spVal']:

            if self.isTree(tree['left']):
                 return  self.treeForeCast(tree['left'],InData,modelEval)
            else:
                return modelEval(model=tree,inData=InData)
        else:
            if self.isTree(tree['right']):
                return  self.treeForeCast(tree['right'],InData,modelEval)
            else:
                return modelEval(model=tree,inData=InData)
    def createForeCast(self,tree,testData,modelEval):
        
        yHat=testData[:,0]*0
        for row in testData:
             yHat[i]=self.treeForeCast(tree,row,modelEval)
        return  yHat   
class treeNode():
     def __init__(self,feat,val,left,right):
          featureToSplitOn=feat
          valueOfSplit=val
          rightBranch=right
          leftBranch=left



