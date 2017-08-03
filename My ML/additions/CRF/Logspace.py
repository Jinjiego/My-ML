import numpy as np
class Logspace:
    def __init__(self):
        self.LOGZERO =np.nan
    def eexp(self,x):
        if np.isnan(x):
            return 0
        else:
            return np.exp(x)
    def eln(self,x):
        if x == 0:
            return self.LOGZERO
        elif x>0:
            return np.log(x)
        else:
            print('Wrong!!!\n\t negative input error')
            return np.nan
    def elnsum(self,elnx,elny):
        if np.isnan(elnx):
            return elny
        elif np.isnan(elny):
            return elnx
        elif elnx > elny:
            return elnx + self.eln(1+np.exp(elny-elnx))
        else:
            return elny + self.eln(1+np.exp(elnx-elny))
    def elnproduct(self,elnx,elny):
        if np.isnan(elnx) or np.isnan(elny):
            return self.LOGZERO
        else:
            return elnx + elny
    def elnmatprod(self,elnx,elny):
        #array([[ 0.]])其size是2
        xsize = np.size(np.shape(elnx))
        ysize = np.size(np.shape(elny))

        if xsize == 1 and ysize == 1:
            r = self.LOGZERO
            for i in range(np.shape(elnx)[0]):
                r = self.elnsum(r,self.elnproduct(elnx[i],elny[i]))
            return r
        elif xsize == 1 and not ysize == 1:
            n = np.shape(elny)[1]
            r = np.zeros(n)
            for i in range(n):
                r[i] = self.elnmatprod(elnx,elny[:,i])
            return r
        elif not xsize == 1 and ysize == 1:
            n = np.shape(elnx)[0]
            r = np.zeros(n)
            for i in range(n):
                r[i] = self.elnmatprod(elnx[i,:],elny)
            return r    
        else:
            m,n= np.shape(elnx)
            p = np.shape(elny)[1]
            r = np.zeros((m,p))
            for i in range(m):
                for j in range(p):
                    r[i][j] = self.elnmatprod(elnx[i,:],elny[:,j])
            return r
    def eexpmat(self,elny):
        expy = np.copy(elny)
        if np.size(np.shape(elny)) == 1:
            for i in range(np.shape(elny)[0]):
                expy[i] = self.eexp(expy[i])
        else:
            for i in range(np.shape(elny)[0]):
                for j in range(np.shape(elny)[1]):
                    expy[i][j] = self.eexp(expy[i][j])
        return expy
    def elnmat(self,x):
        elnx = np.copy(x)
        if np.size(np.shape(x)) == 1:
            for i in range(np.shape(x)[0]):
                elnx[i] = self.eln(x[i])
        else:
            for i in range(np.shape(x)[0]):
                for j in range(np.shape(x)[1]):
                    elnx[i,j] = self.eln(x[i,j])
        return elnx  