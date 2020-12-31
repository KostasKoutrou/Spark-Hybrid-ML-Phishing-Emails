#This program tests the possible performance improvement of having a stacked model
#where a text-based classifier predicts a dataset, and then a properties-based classifier
#used the default properties-based features plus an additional feature which is the
#prediction of the text-based classifier for each example.

import SpyderTest
import curvemetrics
from pyspark.sql.functions import monotonically_increasing_id

#DF, feats, classifiers = SpyderTest.getdata(1000000000000000)

def stacktrain(wordfeat, propfeat, c1, c2, df1, df2):
    result1 = SpyderTest.classtrain(df1, wordfeat, c1.classifier, c1.classifiergrid)
#    result1 = SpyderTest.classtrain(df1, wordfeat, c1.classifier)
    #print(result1.model.bestModel.stages[-1].explainParams())
    #result1.printperformance()
    df2predictions = result1.model.transform(df2)
    cm = df2predictions.select('prediction')
    cm = cm.withColumnRenamed('prediction', 'prevPrediction')
    cm = cm.withColumn("id",monotonically_increasing_id())
    df2 = df2.withColumn("id",monotonically_increasing_id())
    df2 = df2.join(cm, "id")
    df2 = df2.drop("id")
    result2 = SpyderTest.classtrain(df2, propfeat, c2.classifier, c2.classifiergrid)
#    result2 = SpyderTest.classtrain(df2, propfeat, c2.classifier)
#    print(result2.model.bestModel.stages[-1].explainParams())
#    result2.printperformance()
    return result2

def stackcombos(DF, feats, classifiers):
    df1, df2 = DF.randomSplit([0.50, 0.50])
#    wordfeats = [feats[1], feats[2]]
    wordfeats = [feats[2]]
    propfeat = feats[0]
    restclassifiers = classifiers[3:6]
    for wordfeat in wordfeats:
        print("----------------------NEW WORD FEATURE--------------------")
        for c1 in restclassifiers:
            print("################# NEXT C1 ###################")
            for c2 in classifiers:
                wordfeat[-1].setOutputCol('features')
                print("@@@@@@@@@@@@@@@@@@@ NEXT C2 @@@@@@@@@@@@@@@@@")
                result = stacktrain(wordfeat, propfeat, c1, c2, df1, df2)
                result.printperformance()