
from CH03.DTreeTest  import *
from Ch05.Logistic import *
from Ch07.AdaBoost import *
from Ch09.CART import * 
def main():
     dtree=DTreeTest()
     #dtree.test()
     ab=AdaBoost()
     # ab.Invoker()
     cart=CART()
     cart.Invoker()

if __name__=='__main__':
       main()
    