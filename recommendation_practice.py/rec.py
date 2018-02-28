from pyspark.ml.recommendation import ALS
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
import pyspark as ps
import pandas as pd


def load_data():
    sc =ps.SparkContext('local[4]')
    sq = ps.SQLContext(sc)
    train_data = sq.createDataFrame(pd.read_table('data/ratings.dat'))
    test_data = sq.createDataFrame(pd.read_csv('data/dont_use.csv'))
    return train_data, test_data

def fit_best_model(train_data, test_data):


    als = ALS(userCol='user_id', itemCol='joke_id', ratingCol="rating",
              coldStartStrategy='drop', nonnegative=True)
    model = als.fit(train_data)
    params = ParamGridBuilder()\
        .addGrid(als.rank, [11,12,13])\
        .addGrid(als.maxIter, [20,21,22])\
        .build()

    evaluator = RegressionEvaluator(metricName="rmse", labelCol='rating',
                                    predictionCol="prediction")
    tvs = TrainValidationSplit(estimator=als,evaluator=evaluator,estimatorParamMaps=params)
    model = tvs.fit(train_data)
    predictions = model.transform(test_data)
    print(predictions)
    return evaluator.evaluate(predictions)


if __name__ == '__main__':
    train_set, test_set = load_data()
    mod = fit_best_model(train_set, test_set)
    print(mod)
