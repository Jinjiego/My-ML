import math
import numpy as ny
import matplotlib.pyplot as plt
class Logistic(object):
     
     def __init__(self,path):
         self.path=path 
     def LoadDataSet(self,fn):
         fr=open(self.path+fn)
         rows=fr.readlines()
         dataMat=[]
         labelMat=[]
         for row in rows:
             line=row.strip().split()
             dataMat.append([1.0, float(line[0]),float(line[1])])
             labelMat.append(int(line[2]))
         return dataMat,labelMat
     def sigmoid(self,x):
         k=1+ny.exp(-x)
         return ny.mat(1.0/k)

     def gradAscent(self,dataMatin,classlabels):
         # 将整个数据集作为一个整体进行训练，计算量较大 
           dataMatin=ny.mat(dataMatin)
           classlabels=ny.mat(classlabels).transpose()
           N=500
           alpha=0.001
           m,n=ny.shape(dataMatin)
           weights=ny.ones((n,1))
           for i in range(N):
               h=self.sigmoid(dataMatin*weights)
               error=classlabels-h
               #===================================================================
               weights=weights+alpha*dataMatin.transpose()*error
                 # This is the most significant step in which the variable weights 
                 # is adjusted by gradient acending  
               #===================================================================
           return weights
     def stocGradAscent0(self,dataMatin,classlabels):
         # 每一个数据调整一次weight
            dataMatin=ny.mat(dataMatin)
            m,n=ny.shape(dataMatin)
            weights=ny.mat(ny.ones((n,1)))
            alpha=0.001
            for i in range(m):
                   h=self.sigmoid(dataMatin[i]*weights)
                   error=ny.array((classlabels[i]-h)[0])[0][0]
                   temp=dataMatin[i].__copy__().transpose()
                   weights=weights+alpha*error*temp
            return weights
     def plotBestFit(self,weights):
         #plot result figure,including scatter points of data
         # and decision border 
         dataMat,labelMat=self.LoadDataSet('/testSet.txt')
         xcord0=[]; ycord0=[]
         xcord1=[]; ycord1=[]
         for i in range(len(labelMat)):
             if labelMat[i]==0:
                 xcord0.append(dataMat[i][1])
                 ycord0.append(dataMat[i][2])
             else:
                 xcord1.append(dataMat[i][1])
                 ycord1.append(dataMat[i][2])
         fig=plt.figure()
         ax=fig.add_subplot(111)
         ax.scatter(xcord0,ycord0,s=30,c='red',marker='s')
         ax.scatter(xcord1,ycord1,s=30,c='green')
         x=list(ny.arange(-5.0,4.0,0.1))
         y=ny.array((-weights[0]-weights[1]*x)/weights[2])[0].tolist()
         ax.plot(x,y) # what it worth to note is that the function 
                      #plot only accept arguements with type of list 
         plt.xlabel('X1'); plt.ylabel('X2')
         plt.show()
     def Invoker(self):
          self=Logistic('./Ch05')
          dataMat, labelMat=self.LoadDataSet('/testSet.txt')
          weights = self.gradAscent(dataMat, labelMat)
          print(weights)
          # self.plotBestFit(weights)
          weights2=self.stocGradAscent0(dataMat, labelMat)
          print(weights2)
          self.plotBestFit(weights2)

