from utils import get_spark_session
from pyspark.sql.functions import col, explode, lower, trim, split, concat_ws, collect_list
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline


def main():
    spark = get_spark_session("Reddit_ML")

    #===========
    # DATA LOAD
    #===========
    print(f"Loading data...")
    path_all_chunks = "data/chunk*/*.xml"
    path_labels = "data/risk-golden-truth-test.txt"

    #=====================
    # LOAD AND FLATTERING
    #=====================
    df_raw = spark.read.format("xml").option("rowTag", "INDIVIDUAL").load(path_all_chunks)

    df_posts = df_raw.select(col("ID").alias("subject_id"), explode(col("WRITING")).alias("post")) \
        .select(col("subject_id"), lower(trim(col("post.TEXT"))).alias("text"))

    df_labels = spark.read.text(path_labels).select(
        split(col("value"), "\s+").getItem(0).alias("subject_id"),
        split(col("value"), "\s+").getItem(1).cast("int").alias("label")
    )

    #=====================
    # JOIN BETWEEN TABLES
    #=====================
    df_full = df_posts.join(df_labels, on="subject_id", how="inner")

    #==================
    # GROUPING BY USER
    #==================
    print(f"Grouping texts by user...")
    df_grouped = df_full.groupBy("subject_id", "label") \
        .agg(concat_ws(" ", collect_list("text"))
             .alias("all_text"))

    #====================
    # SPLIT AND BALANCING
    #====================
    train_raw, test_data = df_grouped.randomSplit([0.8, 0.2], seed=42)

    print(f"Original data training: {train_raw.count()} users.")

    #===========================
    # UNDERSAMPLING ON TRAINSET
    #===========================
    train_depressed = train_raw.filter(col("label") == 1)
    train_controll = train_raw.filter(col("label") == 0)

    count_dep = train_depressed.count()
    count_ctrl = train_controll.count()

    #====================
    # CHECK DIVISION BY 0
    #====================
    if count_ctrl > 0:
        print(f"Imbalance: {count_dep / count_ctrl * 100:.2f}%")
        fraction = count_dep / count_ctrl
        train_controll_balanced = train_controll.sample(withReplacement=False, fraction=fraction, seed=42)
    else:
        print("Warning: No control users found.")
        train_controll_balanced = train_controll

    #====================================
    # MERGING OF THE TWO BALANCED DATASET
    #====================================
    train_data = train_depressed.union(train_controll_balanced)

    #=============================================
    # TF-IDF PIPELINE CONFIG (Automated cleaning)
    #=============================================
    print(f"Configuring Automated TF-IDF Pipeline...")

    #==========================
    # TOKENIZER (Text -> words)
    #==========================
    tokenizer = RegexTokenizer(inputCol="all_text", outputCol="words", pattern="\\W")

    #===================
    # STANDARD CLEANING
    #===================
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")

    #=======================
    # STATISTICAL CLEANING
    #=======================
    # vocabSize=5000: top 5000 more relevant words
    # minDF=5.0: Ignore words used by fewer than 5 users (typos, rare errors)
    # maxDF=0.75: Ignore words used by more than 75% of users (e.g., "reddit," "post," "today")
    cv = CountVectorizer(
        inputCol="filtered",
        outputCol="raw_features",
        vocabSize=5000,
        minDF=5.0,
        maxDF=0.75
    )

    idf = IDF(inputCol="raw_features", outputCol="features")

    #=====================
    # LOGISTIC REGRESSION
    #=====================
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lr])

    #==========
    # TRAINING
    #==========
    print("Training on balanced dataset...")
    model = pipeline.fit(train_data)

    #============
    # EVALUATION
    #============
    print(f"Testing model on real data (unbalanced)...")
    predictions = model.transform(test_data)

    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                      metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluator_weighted_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                                     metricName="weightedPrecision")
    evaluator_weighted_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",
                                                                  metricName="weightedRecall")

    accuracy = evaluator_acc.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)
    precision = evaluator_weighted_precision.evaluate(predictions)
    recall = evaluator_weighted_recall.evaluate(predictions)

    print("\n --- FINAL RESULTS ---")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"F1-Score:  {f1_score:.2%} <--- Key Metric")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")

    print("\n CONFUSION MATRIX:")
    predictions.groupBy("label", "prediction").count().show()

    #===============
    # SAVING MODEL
    #===============
    print("\nSaving the model...")
    model_path = "models/depression_classifier_model"
    model.write().overwrite().save(model_path)
    print(f"Model correctly saved in: {model_path}")

    #==========================================
    # PRINTING OF THE WORDS CHOSEN BY THE MODEL
    #==========================================
    vocab = model.stages[2].vocabulary  # stage 2: CountVectorizer
    print(f"\nTop 10 most frequent features retained: {vocab[:10]}")
    print(f"Last 10 features retained: {vocab[-10:]}")


if __name__ == "__main__":
    main()