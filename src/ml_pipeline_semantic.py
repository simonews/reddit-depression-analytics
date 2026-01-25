from utils import get_spark_session
from pyspark.sql.functions import col, explode, lower, trim, split, concat_ws, collect_list, size, hour, dayofweek, avg, \
    count, lit
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, Word2Vec
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, StringType, FloatType


def main():
    spark = get_spark_session("Reddit_Project_Master")
    spark.sparkContext.setLogLevel("ERROR")

    print("\n" + "=" * 60)
    print("PIPELINE STARTUP: ML TRAINING + KNOWLEDGE EXTRACTION (SEMANTICS)")
    print("=" * 60 + "\n")

    #===========
    # DATA LOAD
    #===========
    path_all_chunks = "data/chunk*/*.xml"
    path_labels = "data/risk-golden-truth-test.txt"

    print(f"[1/5] Loading and Joining Data...")
    try:
        df_raw = spark.read.format("xml").option("rowTag", "INDIVIDUAL").load(path_all_chunks)
    except Exception as e:
        print(f"Critical error reading data: {e}")
        return

    #============
    # FLATTERING
    #============
    df_posts = df_raw.select(
        col("ID").alias("subject_id"),
        explode(col("WRITING")).alias("post")
    ).select(
        col("subject_id"),
        lower(trim(col("post.TEXT"))).alias("text"),
        col("post.DATE").cast("timestamp").alias("created_at")
    )

    #========
    # LABELS
    #========
    df_labels = spark.read.text(path_labels).select(
        split(col("value"), "\s+").getItem(0).alias("subject_id"),
        split(col("value"), "\s+").getItem(1).cast("int").alias("label")
    )

    df_full = df_posts.join(df_labels, on="subject_id", how="inner")
    df_full.cache()

    #===============================================
    # STATS AGGREGATION (KPI + TIME SPLIT + SCATTER)
    #===============================================
    print("\n[2/5] Running Statistical Aggregations...")

    #=============
    # GLOBAL KPI
    #=============
    df_full_aug = df_full.withColumn("word_count", size(split(col("text"), " ")))
    kpi_df = df_full_aug.agg(
        count("*").alias("total_posts"),
        avg("label").alias("risk_ratio"),
        avg("word_count").alias("avg_length")
    )
    kpi_df.coalesce(1).write.mode("overwrite").csv("data/dashboard_kpi", header=True)

    #==========================================
    # TIME HEATMAP SPLIT (Depressi vs Control)
    #==========================================
    print("   -> Differential Heatmap Generation (Circadian Rhythms)...")

    # Heatmap Depressed (Label 1)
    time_dep = df_full.filter("label=1").withColumn("hour", hour("created_at")) \
        .withColumn("day", dayofweek("created_at")) \
        .groupBy("day", "hour").count().orderBy("day", "hour")
    time_dep.coalesce(1).write.mode("overwrite").parquet("data/dashboard_time_dep.parquet")

    # Heatmap Control (Label 0)
    time_ctrl = df_full.filter("label=0").withColumn("hour", hour("created_at")) \
        .withColumn("day", dayofweek("created_at")) \
        .groupBy("day", "hour").count().orderBy("day", "hour")
    time_ctrl.coalesce(1).write.mode("overwrite").parquet("data/dashboard_time_ctrl.parquet")

    #=============
    # USER SCATTER
    #=============
    user_stats = df_full_aug.groupBy("subject_id", "label") \
        .agg(avg("word_count").alias("avg_len")) \
        .limit(1000)
    user_stats.coalesce(1).write.mode("overwrite").parquet("data/dashboard_scatter.parquet")

    #==========================
    # MACHINE LEARNING TRAINING
    #==========================
    print("\n[3/5] ML Training Preparation (Word2Vec)...")

    df_grouped = df_full.groupBy("subject_id", "label") \
        .agg(concat_ws(" ", collect_list("text")).alias("all_text"))

    train_raw, test_data = df_grouped.randomSplit([0.8, 0.2], seed=42)

    #==========
    # BALANCING
    #==========
    depressed = train_raw.filter("label=1")
    control = train_raw.filter("label=0")
    if control.count() > 0:
        ratio = depressed.count() / control.count()
        train_balanced = depressed.union(control.sample(False, ratio, 42))
    else:
        train_balanced = train_raw

    #==========
    # PIPELINE
    #==========
    print("[4/5] Model Training...")
    tokenizer = RegexTokenizer(inputCol="all_text", outputCol="words", pattern="\\W")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    w2v = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered", outputCol="features")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=30)

    pipeline = Pipeline(stages=[tokenizer, remover, w2v, rf])
    model = pipeline.fit(train_balanced)

    #=====================================
    # EVAL & SAVING METRICS FOR DASHBOARD
    #=====================================
    preds = model.transform(test_data)
    f1 = MulticlassClassificationEvaluator(metricName="f1").evaluate(preds)
    print(f"\n>>> FINAL F1-SCORE: {f1:.2%} <<<\n")

    #=====================
    # SAVING THE F1-SCORE
    #=====================
    metrics_schema = StructType([StructField("metric_name", StringType()), StructField("value", FloatType())])
    metrics_df = spark.createDataFrame([("f1_score", f1)], metrics_schema)
    metrics_df.coalesce(1).write.mode("overwrite").csv("data/dashboard_metrics", header=True)

    model.write().overwrite().save("models/depression_semantic_model")

    #====================
    # SEMANTIC EXTRACTION
    #====================
    print("[5/5] Semantic Graph Generation (Extraction)...")

    w2v_model = model.stages[2]
    seed_words = ["depression", "anxiety", "sadness", "tired", "help", "pain", "empty"]

    schema = StructType([StructField("word", StringType()), StructField("similarity", FloatType())])
    semantic_df = spark.createDataFrame([], schema)

    for seed in seed_words:
        try:
            synonyms = w2v_model.findSynonyms(seed, 10).select("word", col("similarity").cast("float"))
            semantic_df = semantic_df.union(synonyms)
        except Exception:
            pass

    final_semantic_cloud = semantic_df.dropDuplicates(["word"]).orderBy(col("similarity").desc()).limit(50)
    final_semantic_cloud.coalesce(1).write.mode("overwrite").csv("data/dashboard_semantic_words", header=True)

    print("[COMPLETED] Job Finished.")


if __name__ == "__main__":
    main()