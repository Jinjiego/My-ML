from CH03.trees import *
class DTreeTest(object):
    """description of class"""
    # The process to classify data with decision tree includes follow moudles: 
    # splitData()
    # chooseBestFeasure()
    # calculateShannot
    def test(self):
            
            ts=trees()
            DataSet,label=ts.creatDataSet()
            ShannonEnt=ts.calcShannonEnt(DataSet)
            print(ShannonEnt)
            t_tree=ts.createTree(DataSet,label)
            lenses,labels=ts.renderData()
            #print(t_tree)
            t_tree1=ts.createTree(lenses,labels)
            ts.createPlot(t_tree1)
            print( ts.retrieveTree(1))



