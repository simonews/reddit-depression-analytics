from utils import get_spark_session
from pyspark.sql.functions import col, explode, lower, trim, split, concat_ws, collect_list
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

def main():
    spark = get_spark_session("Reddit_ML")

    # ---1.Caricamento dei dati ---
    print(f"Loading data...")
    path_all_chunks = "data/chunk*/*.xml"
    path_labels = "data/risk-golden-truth-test.txt"

    #Caricamento e flattering
    df_raw = spark.read.format("xml").option("rowTag", "INDIVIDUAL").load(path_all_chunks)

    df_posts = df_raw.select(col("ID").alias("subject_id"), explode(col("WRITING")).alias("post")) \
        .select(col("subject_id"), lower(trim(col("post.TEXT"))).alias("text"))

    df_labels = spark.read.text(path_labels).select(
        split(col("value"), "\s+").getItem(0).alias("subject_id"),
        split(col("value"), "\s+").getItem(1).cast("int").alias("label")
    )

    #Join tra le tabelle
    df_full = df_posts.join(df_labels, on="subject_id", how="inner")

    # ---2. Raggruppamento per utente ---
    print(f"Grouping texts by user...")
    df_grouped = df_full.groupBy("subject_id", "label") \
                        .agg(concat_ws(" ", collect_list("text"))
                        .alias("all_text"))

    # ---3. Split e balancing ---
    train_raw, test_data = df_grouped.randomSplit([0.8, 0.2], seed=42)

    print(f"Original data training: {train_raw.count()} users.")

    #Undersampling solo sul training
    train_depressed = train_raw.filter(col("label") == 1)
    train_controll = train_raw.filter(col("label") == 0)

    count_dep = train_depressed.count()
    count_ctrl = train_controll.count()
    print(f"Imbalance: {count_dep / count_ctrl * 100}%")

    fraction = count_dep / count_ctrl
    train_controll_balanced = train_controll.sample(withReplacement=False, fraction=fraction, seed=42)

    #Unione dei due dataset bilanciati
    train_data = train_depressed.union(train_controll_balanced)

    print(f"TF-IDF Pipeline Configuration + Logistic Regression...")

    #Stopwords
    custom_stop_words = [
        "the", "a", "an", "and", "or", "but", "if", "because", "as", "what",
        "when", "where", "how", "which", "who", "whom", "why", "from", "to",
        "of", "in", "on", "at", "by", "for", "with", "about", "into", "through",
        "during", "before", "after", "above", "below", "up", "down", "out", "off",
        "over", "under", "again", "further", "then", "once", "here", "there",
        "is", "are", "was", "were", "be", "been", "being", "am", "im", "ive",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "can", "could", "shall", "should", "will", "would", "may", "might", "must",
        "i", "me", "my", "myself", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "this", "that", "these", "those",
        "just", "really", "very", "so", "like", "much", "many", "some", "any", "no", "not",
        "only", "own", "same", "too", "than", "also", "even", "back", "get", "go", "going", "got",
        "well", "way", "made", "make", "know", "think", "thing", "things", "see", "say", "said",
        "want", "need", "good", "bad", "better", "best", "still", "never", "ever", "last",
        "right", "left", "sure", "actually", "maybe", "probably", "one", "two", "people",
        "someone", "something", "time", "years", "day", "days", "life", "look", "use",
        "all", "more", "now", "first", "since", "while", "lot", "little", "few", "take",
        "http", "https", "www", "com", "reddit", "subreddit", "post", "posts", "comment", "comments",
        "img", "png", "jpg", "html", "php", "url", "link", "deleted", "removed",
        "don", "didn", "doesn", "isn", "wasn", "weren", "hasn", "haven", "hadn",
        "won", "wouldn", "couldn", "shouldn", "cant", "cannot", "wont", "arent"
    ]

    tokenizer = RegexTokenizer(inputCol="all_text", outputCol="words", pattern="\\W")
    remover = StopWordsRemover(inputCol="words", outputCol="filtered", stopWords=custom_stop_words)
    cv = CountVectorizer(inputCol="filtered", outputCol="raw_features", vocabSize=3000, minDF=2.0)
    idf = IDF(inputCol="raw_features", outputCol="features")
    lr = LogisticRegression(featuresCol="features", labelCol="label", maxIter=20)

    pipeline = Pipeline(stages=[tokenizer, remover, cv, idf, lr])

    # ---5.Addestramento ---
    print("Training on balanced dataset...")
    model = pipeline.fit(train_data)

    # valutazione del modello
    print(f"Testing model on real data (unbalanced)...")
    predictions = model.transform(test_data)

    evaluator_acc = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="accuracy")
    evaluator_f1 = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
    evaluator_weighted_precision = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="weightedPrecision")
    evaluator_weighted_recall = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction",metricName="weightedRecall")

    accuracy = evaluator_acc.evaluate(predictions)
    f1_score = evaluator_f1.evaluate(predictions)
    precision = evaluator_weighted_precision.evaluate(predictions)
    recall = evaluator_weighted_recall.evaluate(predictions)

    print("\n --- FINAL RESULTS ---")
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"F1-Score:  {f1_score:.2%} <---")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")

    print("\n CONFUSION MATRIX:")
    # label 1 = Depresso, prediction 1 = predetto depresso
    predictions.groupBy("label", "prediction").count().show()

    print("LEGEND:")
    print("Label 1, Pred 1 -> TRUE POSITIVES (Depressed found correct)")
    print("Label 1, Pred 0 -> FALSE NEGATIVES (Depressed not found)")
    print("Label 0, Pred 1 -> FALSE POSITIVES (Healthy people reported as depressed)")
    print("Label 0, Pred 0 -> TRUE NEGATIVES (Healthy found correct)")

if __name__ == "__main__":
    main()