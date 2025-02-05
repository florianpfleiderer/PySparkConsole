from pyspark.sql import SparkSession

DATA_PATH = "data/pupil-absence-in-schools-in-england_2018-19/Absence_3term201819_nat_reg_la_sch.csv"

spark = SparkSession.builder.master("local[*]").appName("MySpark").getOrCreate()

data = spark.read.format("csv").load(DATA_PATH)

data.show()
