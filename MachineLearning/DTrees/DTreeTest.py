from DTrees.trees import *
class DTreeTest(object):
    """description of class"""
    def test(self):
            ts=trees()
            DataSet,label=ts.creatDataSet()
            #ShannonEnt=ts.calcShannonEnt(DataSet)
            #print(ShannonEnt)
            #lenses,labels=ts.renderData()
            t_tree=ts.createTree(DataSet,label)
            #print(t_tree)
           #ts.createPlot(t_tree)
            print( ts.retrieveTree(1))



