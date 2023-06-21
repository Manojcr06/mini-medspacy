from pyspark.sql import SparkSession
from sparknlp.annotator import Tokenizer, NerDLModel, NerConverter
from pyspark.ml import Pipeline
from sparknlp.common import *
from sparknlp.base import DocumentAssembler
import pandas as pd

# Create a SparkSession
spark = SparkSession.builder \
    .appName("Spark NLP")\
    .master("local[*]")\
    .config("spark.driver.memory","16G")\
    .config("spark.driver.maxResultSize", "0") \
    .config("spark.kryoserializer.buffer.max", "2000M")\
    .config("spark.jars.packages", "com.johnsnowlabs.nlp:spark-nlp_2.12:4.4.3")\
    .getOrCreate()

# Assuming you have a DataFrame called 'dfo' with a column named 'text' containing the sentences
dfo = spark.createDataFrame(pd.DataFrame({'text': [
"Atrial Fibrillation causes cancer across all the genetics and ages. It's widespread in the globe as caused chaos.",
"Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
"This is another example sentence with a Medical Term like Heart Disease."
]}))

# Load the Spark NLP pipeline
documentAssembler = DocumentAssembler().setInputCol("text").setOutputCol("document")
tokenizer = Tokenizer().setInputCols(["document"]).setOutputCol("token")
nerModel = NerDLModel.pretrained("ner_dl_medical", "en", "clinical/models").setInputCols(["document", "token"]).setOutputCol("ner")
nerConverter = NerConverter().setInputCols(["document", "token", "ner"]).setOutputCol("ner_span")
nerPipeline = Pipeline(stages=[documentAssembler, tokenizer, nerModel, nerConverter])

# Fit the pipeline to the data
pipelineModel = nerPipeline.fit(dfo)

# Transform the data to extract medical terms
extractedData = pipelineModel.transform(dfo)

# Select the relevant columns from the transformed data
result = extractedData.select("text", "ner_span.result").toPandas()

# Rename the column to "medical_terms"
result = result.rename(columns={"result": "medical_terms"})

# Save the results to a CSV file
result.to_csv("medical_terms.csv", index=False)