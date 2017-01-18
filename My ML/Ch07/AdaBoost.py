import numpy as ny
class AdaBoost(object):
    """description of class"""
    def loadsimpleData(self):
        datMat = ny.matrix([[1.0,2.1],
                          [2.0,1.1],
                          [1.3,1.0],
                          [1.0,1.0],
                          [2.0,1.1]])
        classLabels = [1.0,1.0,-1.0,-1.0,1.0]
        return datMat,classLabels
    def buildStump(self,datMatIn,classLabels,D):
        datMat = ny.mat(datMatIn)
        labelMat = ny.mat(classLabels).T
        m,n = ny.shape(datMat)
        numStep = 10
        BestStump = {}
        MinErr = ny.Inf
        bestClassEst = ny.mat(ny.zeros((m,1)))
        for i in range(n):
             rangeMin = datMat[:,i].min()
             rangeMax = datMat[:,i].max()   
             stepSize = (rangeMax - rangeMin) / numStep
             for j in range(-1,numStep + 1):
                  for inequal in ['lt','gt']:
                         threshval = rangeMin + float(j) * stepSize
                         predictedVals = self.stumpClassfy(datMat,i,threshval,inequal)
                         errAttr = ny.mat(ny.ones((m,1)))
                         errAttr[predictedVals == labelMat] = 0
                         error = D.T * errAttr
                         print("split: dim %d, thresh %.3f, thresh inequal: %s\
                                the weighted error is %f ." % (i,threshval,inequal,error))
                         if error < MinErr:
                              MinErr = error
                              bestClassEst = predictedVals.copy()
                              BestStump["dim"] = i
                              BestStump["threshval"] = threshval
                              BestStump["ineq"] = inequal
        return BestStump,MinErr,bestClassEst
    #==============build stump tree for classifing=============================
    def stumpClassfy(self,datMatIn,dimen,threshval,threshIneq):
        retArray = ny.ones((ny.shape(datMatIn)[0],1))
        if  threshIneq == 'lt':
            retArray[datMatIn[:,dimen] <= threshval] = -1.0
        else:
            retArray[datMatIn[:,dimen] > threshval] = -1.0
        return retArray    
    def Invoker(self): 
          
          datMat,classLabels = self.loadsimpleData()     
          m,n = ny.shape(datMat)
          D = ny.mat(ny.ones((m,1))) / m                   
          print(self.buildStump(datMat,classLabels,D))
          self.adaBoostTrainDS(datMat,classLabels,9)

          datMat,labelMat = self.loadDataSet('./Ch07/horseColicTraining2.txt')
          classifierArray = self.adaBoostTrainDS(datMat,labelMat,10)
          datMat,labelMat = self.loadDataSet('./Ch07/horseColicTest2.txt')
          Predicted10 = self.adaClassify(datMat,classifierArray)

          errArr = ny.mat(ny.ones((len(datMat),1)))
          print(errArr[Predicted10 != ny.mat(labelMat).T].sum())
    #===============Train adaBoost, Construct a series of weak classifiers and
    #weights=============================
    def adaBoostTrainDS(self,dataArr,classLabels,numIt=40):
         #The algorithm of adaBoost decision stump
         weakClassArr = []
         m = ny.shape(dataArr)[0]
         D = ny.mat(ny.ones((m,1)) / m)
         aggClassEst = ny.mat(ny.zeros((m,1)))
         for i in range(numIt):
               ####-----0 build a weak classifier
               bestStump,error,classEst = self.buildStump(dataArr,classLabels,D)
               ####-----1 calculate the weight of this
               ####classfier-----------------------------
               alpha = float(0.5 * ny.log((1.0 - error) / max(error,1e-16)))
               bestStump['alpha'] = alpha
               weakClassArr.append(bestStump) #Record the weak classsifier
               ####-----2 calculate the vector D,in which D[i] denotes the
               ####weight of the i-th sample
               #expon=ny.multiply(-1*alpha*ny.mat(classLabels).T,classEst)
               expon = -1 * alpha * ny.reshape(ny.array(classLabels),(m,1)) * \
                         ny.reshape(ny.array(classEst),(m,1)) 
               D = ny.reshape(ny.array(D),(m,1)) * ny.exp(expon) 
               D = D / D.sum()
               print("D:",D.T) 
               print("classEst:",classEst.T)
               aggClassEst+= alpha * classEst  
               print("aggClassEst:",aggClassEst)
               #It is used for recording the classfy result of existing
               #classfier in weakClassArr
               aggErrors = ny.reshape(ny.array(ny.sign(aggClassEst).T != classLabels),(m,1)) * \
                   ny.array(ny.ones((m,1)))  
               #calculate the error of existing classifier
               errorRate = aggErrors.sum() / m  
               print("total error:",errorRate,"\n")
               if errorRate == 0.0 :
                     break
         return weakClassArr

    def adaClassify(self,dat2Class,classifierArr):
           dataMatrix = ny.mat(dat2Class)
           m,n = ny.shape(dataMatrix)
           classEst = ny.zeros((m,1))
           for i in range(len(classifierArr)):
                classEst = self.stumpClassfy(dataMatrix,\
                                 classifierArr[i]['dim'],\
                                 classifierArr[i]['threshval'],\
                                 classifierArr[i]['ineq'])
                classEst+=classifierArr[i]['alpha'] * classEst
                print(classEst)
           return ny.sign(classEst)
    def loadDataSet(self,fn):
         datMat = []
         labelMat = []
         try:
             fr = open(fn)
             Rows = fr.readlines()
         except Exception:
             print('File read error!')
         else: #If there aren't have any exception
             numFeat = len(Rows[1].strip().split('\t'))
             for line in Rows:
                  lines = line.strip().split('\t')
                  temp = []
                  for e in lines[:-1]: 
                      temp.append(float(e))
                  datMat.append(temp)
                  labelMat.append(float(lines[-1]))
                   
         return datMat,labelMat

         
        









