
def draw(f,xrange):
    step=(xrange[1]-xrange[0])/100
    
def gradient(x0,rate,err,f,pf):
    #f(x)=-x^2+3x-1
    x1=x0+err
    while(abs(x1-x0)>=err):
        x0=x1
        x1=x0+rate*pf(x0)  #�����ݶ�����
        print(x1)

    return f(x0)

if __name__=="__main__":
   a= gradient(0,-0.01,1e-5,lambda x:x^2+3*x-1,lambda x:2*x+3)
   print(a)
    
