
from CH03.DTreeTest  import *
from Ch05.Logistic import *
from Ch07.AdaBoost import *
from Ch08.Regression import *
from Ch09.CART import *
from Ch11.Apriori import  *
from Ch12.FP_growth import *
def main():
     rgess=Regression()
     rgess.debug()

     fp=FP_growth()
     fp.debug()

     apr=Apriori()
     apr.debug()

     dtree=DTreeTest()
     #dtree.test()
     ab=AdaBoost()
     # ab.Invoker()
     cart=CART()
     cart.Invoker()

if __name__=='__main__':
       main()
    