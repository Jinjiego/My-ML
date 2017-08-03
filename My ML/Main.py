
from DecisionTree.DTreeTest  import *
from LogisticRegression.Logistic import *
from Boost.AdaBoost import *
from LinearRegression.Regression import *
from CART.CART import *
from Apriori.Apriori import  *
from FpGrowth.FP_growth import *
from additions.CRF.crf_test import *
def main():
     
     fp=FP_growth()
     fp.debug()

     test();

     cart=CART()       #CART 
     cart.Invoker()

     dtree=DTreeTest() #ID3
     dtree.test()  


     rgess=Regression()
     rgess.debug()

     apr=Apriori()
     apr.debug()

     ab=AdaBoost()
     # ab.Invoker()

if __name__=='__main__':
       main()
    