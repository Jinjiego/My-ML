from Bayes.Bayes import  *
class BayesTest(object):
      """description of class"""
      def  __init__(self):
              pass
      def test(self):
              bys=Bayes()

              postingList,classVect=bys.loadDataSet()
              print("postingList:\n",postingList)
              vocabList=bys.createVocabraryList(postingList)
              vocabListVect=bys.setOfWord2Vex(vocabList,postingList[0])
              print("vocabList:(",len(vocabList),")\n",str(vocabList))
              print(vocabListVect)
              
              trainMat=bys.renderTrain()
              print("trainMat:")
              for i in range(len(trainMat)):
                    print(trainMat[i])
              p0Vect,p1Vect,pAbusive=bys.trainNB0(trainMat,classVect)
              print("p0Vect:\n",p0Vect)
              print("p1Vect:\n",p1Vect)
              print("pAbusive:\n",pAbusive)
              #-------------------------------------------------------------------------------------------
              testDoc=['love', 'my', 'dalmation']
              testDocVect=bys.setOfWord2Vex(vocabList,testDoc)
              print("testDoc has been classified to ",bys.classifyNB0(testDocVect,p0Vect,p1Vect,pAbusive) )

              testDoc=['stupid', 'garbage']
              testDocVect=bys.setOfWord2Vex(vocabList,testDoc)
              print("testDoc has been classified to ",bys.classifyNB0(testDocVect,p0Vect,p1Vect,pAbusive) )

