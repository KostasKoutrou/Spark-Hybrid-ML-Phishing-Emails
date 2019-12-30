#This program tests the possible performance improvement of having a stacked model
#where a text-based classifier predicts a dataset, and then a properties-based classifier
#used the default properties-based features plus an additional feature which is the
#prediction of the text-based classifier for each example.

import SpyderTest
import curvemetrics

DF, feats, classifiers = SpyderTest.getdata(5000)

df1, df2 = DF.randomSplit([0.50, 0.50])

result1 = SpyderTest.classtrain(df1, feats[2], classifiers[0].classifier, classifiers[0].classifiergrid)
print(result1.model.bestModel.stages[-1].explainParams())
result1.printperformance()

df2predictions = result1.model.transform(df2)
from pyspark.sql.functions import monotonically_increasing_id
cm = df2predictions.select('prediction')
cm = cm.withColumnRenamed('prediction', 'prevPrediction')
cm = cm.withColumn("id",monotonically_increasing_id())
df2 = df2.withColumn("id",monotonically_increasing_id())
df2 = df2.join(cm, "id")
df2 = df2.drop("id")

result2 = SpyderTest.classtrain(df2, SpyderTest.propstages(df2), classifiers[0].classifier, classifiers[0].classifiergrid)
print(result2.model.bestModel.stages[-1].explainParams())
result2.printperformance()


featlist = [result1, result2]
curvemetrics.plotCurves(featlist)