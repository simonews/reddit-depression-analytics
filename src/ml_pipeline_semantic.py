from utils import get_spark_session
from pyspark.sql.functions import col, explode, lower, trim, split, concat_ws, collect_list, size, hour, dayofweek, avg, \
    count, lit, udf, log2, sum as _sum
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

    print(f"[1/6] Loading and Joining Data...")
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
    print("\n[2/6] Running Statistical Aggregations...")

    #=============
    # GLOBAL KPI
    #=============
    ("   -> Calculating Global KPI...")
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
    print("   -> Define the user scatter...")
    user_stats = df_full_aug.groupBy("subject_id", "label") \
        .agg(avg("word_count").alias("avg_len")) \
        .limit(1000)
    user_stats.coalesce(1).write.mode("overwrite").parquet("data/dashboard_scatter.parquet")


    # =========================================================
    # D. SHANNON ENTROPY (Cognitive Rigidity Calculation)
    # =========================================================
    print("   -> Calculating Shannon Entropy (Linguistic Complexity)...")

    # Simple tokenization
    df_words = df_full.select("subject_id", "label", explode(split(col("text"), " ")).alias("word")) \
        .filter(col("word") != "")

    # Count words occurrences for every user (n_i)
    word_counts = df_words.groupBy("subject_id", "label", "word").count().withColumnRenamed("count", "n_i")

    # words total for every user (N)
    total_counts = df_words.groupBy("subject_id").count().withColumnRenamed("count", "N")

    # Entropy: -SUM( (n_i/N) * log2(n_i/N) )
    entropy_df = word_counts.join(total_counts, "subject_id") \
        .withColumn("p_i", col("n_i") / col("N")) \
        .withColumn("entropy_contribution", col("p_i") * log2("p_i")) \
        .groupBy("subject_id", "label") \
        .agg((-1 * _sum("entropy_contribution")).alias("shannon_entropy")) \
        .limit(2000)  # Limited for dash performance

    entropy_df.coalesce(1).write.mode("overwrite").parquet("data/dashboard_entropy.parquet")

    #==========================
    # MACHINE LEARNING TRAINING
    #==========================
    print("\n[3/6] ML Training Preparation (Word2Vec)...")

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
    print("[4/6] Model Training...")
    tokenizer = RegexTokenizer(inputCol="all_text", outputCol="words", pattern="\\W")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered")
    w2v = Word2Vec(vectorSize=100, minCount=5, inputCol="filtered", outputCol="features")
    rf = RandomForestClassifier(labelCol="label", featuresCol="features", numTrees=30)

    pipeline = Pipeline(stages=[tokenizer, remover, w2v, rf])
    model = pipeline.fit(train_balanced)

    #=====================================
    # EVAL & SAVING METRICS FOR DASHBOARD
    #=====================================
    print("[5/6] Calculating Advanced Metrics (Confusion Matrix & Spider Chart Data)...")
    preds = model.transform(test_data)

    # Extended metrichs for Spider Chart
    eval_f1 = MulticlassClassificationEvaluator(metricName="f1")
    eval_acc = MulticlassClassificationEvaluator(metricName="accuracy")
    eval_prec = MulticlassClassificationEvaluator(metricName="weightedPrecision")
    eval_rec = MulticlassClassificationEvaluator(metricName="weightedRecall")

    f1 = eval_f1.evaluate(preds)
    acc = eval_acc.evaluate(preds)
    prec = eval_prec.evaluate(preds)
    rec = eval_rec.evaluate(preds)

    print(f"\n>>> METRICS: F1={f1:.2%}, ACC={acc:.2%}, PREC={prec:.2%}, REC={rec:.2%} <<<\n")

    # Saving
    metrics_data = [
        ("F1-Score", f1),
        ("Accuracy", acc),
        ("Precision", prec),
        ("Recall", rec)
    ]
    metrics_schema = StructType([StructField("metric_name", StringType()), StructField("value", FloatType())])
    metrics_df = spark.createDataFrame(metrics_data, metrics_schema)
    metrics_df.coalesce(1).write.mode("overwrite").csv("data/dashboard_metrics", header=True)

    # Confusion Matrix (Label vs Prediction)
    cm_df = preds.groupBy("label", "prediction").count()
    cm_df.coalesce(1).write.mode("overwrite").csv("data/dashboard_confusion_matrix", header=True)

    # ROC CURVE DATA
    get_prob = udf(lambda v: float(v[1]), FloatType())

    roc_df = preds.select(
        col("label"),
        get_prob("probability").alias("score")
    )
    roc_df.coalesce(1).write.mode("overwrite").csv("data/dashboard_roc_data", header=True)

    model.write().overwrite().save("artifacts/models/depression_semantic_model")

    # ================================================
    # SEMANTIC EXTRACTION (GRAPH AND BAR CHART)
    # ================================================
    print("[6/6] Semantic knowledge extraction (Graph & List)...")

    w2v_model = model.stages[2]
    seed_words = ["depression", "anxiety", "insomnia", "suicide", "hopeless", "meds", "pain"]

    # Graph schema: source -> target
    schema_graph = StructType([
        StructField("source", StringType()),
        StructField("target", StringType()),
        StructField("similarity", FloatType())
    ])
    semantic_graph_df = spark.createDataFrame([], schema_graph)

    for seed in seed_words:
        try:
            # Find synonyms
            synonyms = w2v_model.findSynonyms(seed, 8) \
                .select(lit(seed).alias("source"), col("word").alias("target"), col("similarity").cast("float"))

            semantic_graph_df = semantic_graph_df.union(synonyms)
        except Exception:
            pass

    # Saving dataset for the graph
    semantic_graph_df.coalesce(1).write.mode("overwrite").csv("data/dashboard_semantic_graph", header=True)

    # Saving dataset for the list
    # Only target e similarity
    semantic_words_df = semantic_graph_df.select(col("target").alias("word"), "similarity") \
        .dropDuplicates(["word"]) \
        .orderBy(col("similarity").desc()) \
        .limit(20)

    semantic_words_df.coalesce(1).write.mode("overwrite").csv("data/dashboard_semantic_words", header=True)

    print("[COMPLETED] Job Finished.")


if __name__ == "__main__":
    main()