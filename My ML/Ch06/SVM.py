import numpy as ny
class SVM(object):
    """description of class"""
    def __init__(self,path):
      self.path=path
    def LoadDataSet(self,filename):
        dataMat=[]; labelMat=[]
        fr=open(self.path+filename)
        for row in fr.readlines():
           row.strip().split('\t')
           dataMat.append([float(row[0]),float(row[1])])
           labelMat.append(float(row[-1])) 
        return dataMat,labelMat
    def selectJRand(self,i,m):
        j=i
        while(j==i):
            j=int(random.uniform(0,m))
        return j
    def clipAlpha(aj,H,L):
        if aj>H:
            aj=H
        if aj<L:
            aj=L
        return aj
    def smosimple(self,dataMatIn,classlabels,C,toler,maxIter):
        dataMatrix=ny.mat(dataMatIn);labelMat=ny.mat(classlabels) #construct data matrixs
        b=0; m,n=shape(dataMatrix)
        alphas=ny.zeros((n,1))  #initialize alpha as a zero vector
        iter=0
        while(iter<maxIter): 
            alphaPairsChanged=0



         
            


