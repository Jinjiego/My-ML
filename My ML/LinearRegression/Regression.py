import numpy as np
import matplotlib.pyplot as plt
class Regression(object):
     def __init__(self):
          pass
     def loadDataSet(self,file):
           data=open(file).readlines()
           dataMat=[];labelMat=[]
           numFeat=len(data[0])-1
           for line in data:
                curLine=line.strip().split('\t')
                curLine=list(map(float,curLine))
                dataMat.append(curLine[:-1])
                labelMat.append(curLine[-1])
           return dataMat,labelMat
     def StandRegres(self,X,Y):
          xMat=np.mat(X);
          yMat=np.mat(Y).T
          xTx=xMat.T*xMat
          if np.linalg.det(xTx)==0.0:
               print('x.T*x=0!')
               return ;
          ws=xTx.I*(xMat.T*yMat)
          if len(X[0])==2:
               plt.scatter(xMat[:,1],yMat)
               plt.hold(True)
               xx=xMat.copy()
               xx.sort(0)
               plt.plot(xx[:,1].tolist(),xx*ws)
               plt.title("standard linear regression")
               plt.show()

          return ws   
     def LWLR(self,testPoint,X,Y,k=1.0):
         # testPoint,X,Y are list 
         testPoint=np.mat(testPoint)
         xMat=np.mat(X);yMat=np.mat(Y).T
         m=np.shape(xMat)[0]
         W=np.mat(np.eye(m))
         for j in range(m):
             diffMat=testPoint-xMat[j,:]
             W[j,j]=np.exp(diffMat*diffMat.T/(-2.0*k**2))
         xTWx=xMat.T*(W*xMat)
         if np.linalg.det(xTWx)==0.0:
              print("det(xTWx)=0 !")
              return 
         ws=xTWx.I*(xMat.T*(W*yMat))
         return testPoint*ws
     def LWLRTest(self,testArr,X,Y,k=1.0):
           m=len(testArr)
           yHat=np.zeros(m)
           i=0
           for p in testArr:
                yHat[i]=self.LWLR(p,X,Y,k)
                i+=1
           if len(X[0])==2:
                 xMat=np.mat(X)
                 srtInd=xMat.argsort(0)[:,1]
                 xsorted=xMat[srtInd][:,0,:]
                 fig=plt.scatter(np.mat(X)[:,1],np.mat(Y))
                 
                 plt.hold(True)
                 plt.plot(xsorted[:,1],yHat[srtInd])
                 plt.title("LWLR")
                 plt.show()
                
           return yHat  
     def rssError(self,yArr,yHatArr):
           return ((yArr-yHatArr)**2).sum()     
     def ridgeRegres(self,xMat,yMat,lam=0.2):
           pass

          
                
     def debug(self):
         X,Y= self.loadDataSet('./Ch08/ex0.txt')
         #ws= self.StandRegres(X,Y)

         self.LWLRTest(X,X,Y,0.001)

         a=0 