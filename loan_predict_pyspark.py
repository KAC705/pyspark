import os
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.sql.functions import when, col




base_dir = os.path.abspath(os.path.dirname(__file__))

input_path = f"file://{os.path.join(base_dir, 'loan_data.csv')}"
output_path = os.path.join(base_dir, "output.txt")

spark = SparkSession.builder.appName("Loan_predict").getOrCreate()

df = spark.read.csv(input_path, header=True, inferSchema=True)

# Before I start preprocessing/dealing with Nan, I'm going to drop columns that I believe may lead to unintetntional bias.
# I recognize that these may be influencing factors, but historical data being used to train my model is more likely to contain bias for these catagories.
# Also dropping Loan ID since it is just an identifier and not data relating to loan approval. 

df = df.drop('Gender' , 'Married' , 'Property_Area' , 'Dependents' , 'Education' , 'Loan_ID')

# Dependents, Self_Employed , Loan_Amount , Loan_Amount_term , and Credit_History all have missing values.
# After calculating the percentage of missings for each column I saw that each column had less than 10% missings so I decided to use dropna.
# I then checked to make sure this did not inadvertently delete a majority of my rows. 
# With no missing values, this data set still retains 82% of the data, which is borderline, but I think will be OK for this project. 

df = df.dropna()

# Discretize CoApplicantIncome 
# Im going to split this into if there is a coapplicant (income present) or there is no coapplicant (no income present) to avoid the incomes of 0 for no coapplicant skewing data. 

df['HasCoapplicant'] = (df['CoapplicantIncome'] > 0 ).astype(int)
df = df.drop(columns = ['CoapplicantIncome'])

# Discretize 
# I'm going to make even bin width until I reach the end of the majority of the data
# Then I will make one bin for all the outliers so I can stil include them without the data being skewed. 

df = df.withColumn("HasCoapplicant", when(col("CoapplicantIncome") > 0, 1).otherwise(0)).drop("CoapplicantIncome")

# Binned ApplicantIncome
df = df.withColumn("bined_ApplicantIncome", 
    when(col("ApplicantIncome") <= 1000, 0)
    .when(col("ApplicantIncome") <= 2000, 1)
    .when(col("ApplicantIncome") <= 3000, 2)
    .when(col("ApplicantIncome") <= 4000, 3)
    .when(col("ApplicantIncome") <= 5000, 4)
    .when(col("ApplicantIncome") <= 6000, 5)
    .when(col("ApplicantIncome") <= 7000, 6)
    .when(col("ApplicantIncome") <= 8000, 7)
    .when(col("ApplicantIncome") <= 9000, 8)
    .when(col("ApplicantIncome") <= 10000, 9)
    .otherwise(10)
).drop("ApplicantIncome")

# Binned LoanAmount
df = df.withColumn("Binned_LoanAmount", 
    when(col("LoanAmount") <= 100, 0)
    .when(col("LoanAmount") <= 200, 1)
    .when(col("LoanAmount") <= 300, 2)
    .when(col("LoanAmount") <= 400, 3)
    .otherwise(4)
).drop("LoanAmount")

# Binned Loan Term
df = df.withColumn("binned_LoanAmountTerm", 
    when(col("Loan_Amount_Term") <= 100, 0)
    .when(col("Loan_Amount_Term") <= 200, 1)
    .when(col("Loan_Amount_Term") <= 300, 2)
    .when(col("Loan_Amount_Term") <= 400, 3)
    .otherwise(4)
).drop("Loan_Amount_Term")

# Encoding

Self_Employeed_Indexer = StringIndexer(inputCol = 'Self_Employed' , outputCol = 'Self_Employed_indexed')
Loan_Status_Indexer = StringIndexer(inputCol = 'Loan_Status' , outputCol = 'Loan_Status_indexed')

# ML models
# For my 3 models I will run 3 random forest models with number of trees being 100 , 500 , and 1000. 
# Each model will be evaluated with train test split of 75/25 and accuracy score will be computed. 

assembler = VectorAssembler( inputCols = ['Self_Employed_indexed', 'Credit_History', 'HasCoapplicant', 'bined_ApplicantIncome', 'binned_LoanAmountTerm', 'Binned_LoanAmount'] ,
                            outputCol = 'features')

train_data , test_data = df.randomSplit([.75 , .25])

evaluator = MulticlassClassificationEvaluator(labelCol="Loan_Status_indexed", predictionCol="prediction", metricName="accuracy")


# ML 1: rf100

rf100 = RandomForestClassifier(labelCol = "Loan_Status_indexed" , featuresCol = 'features' , numTrees = 100)

pipeline = Pipeline(stages = [Self_Employeed_Indexer , Loan_Status_Indexer ,  assembler , rf100])

model = pipeline.fit(train_data)

predictions = model.transform(test_data)

accuracy = evaluator.evaluate(predictions)

with open(output_path, "w") as f:
    f.write(f"Random Forest Classification Accuracy: {accuracy:.4f}\n")

# ML 2: rf500

rf500 = RandomForestClassifier(labelCol = "Loan_Status_indexed" , featuresCol = 'features' , numTrees = 500)

pipeline = Pipeline(stages = [Self_Employeed_Indexer , Loan_Status_Indexer ,  assembler , rf500])

model = pipeline.fit(train_data)

predictions = model.transform(test_data)

accuracy = evaluator.evaluate(predictions)

with open(output_path, "a") as f:
    f.write(f"Random Forest Classification Accuracy: {accuracy:.4f}\n")

# ML 3: rf1000

rf1000 = RandomForestClassifier(labelCol = "Loan_Status_indexed" , featuresCol = 'features' , numTrees = 1000)

pipeline = Pipeline(stages = [Self_Employeed_Indexer , Loan_Status_Indexer ,  assembler , rf1000])

model = pipeline.fit(train_data)

predictions = model.transform(test_data)

accuracy = evaluator.evaluate(predictions)

with open(output_path, "a") as f:
    f.write(f"Random Forest Classification Accuracy: {accuracy:.4f}\n")