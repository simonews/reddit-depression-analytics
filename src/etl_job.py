from utils import get_spark_session
from pyspark.sql.functions import explode, col, trim, split, lower, regexp_replace
from pyspark.sql.types import IntegerType
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover

def main():
    # --- Avvio spark ---
    spark = get_spark_session("Reddit_Full_NLp_ETL")

    #Definiamo i percorsi dati
    path_all_chunks = "data/chunk*/*.xml"
    path_labels = "data/risk-golden-truth-test.txt"

    print("\n Starting data analysis...")

    # --- Lettura dei post (XML) ---
    print(f"Reading all XML chunks from: {path_all_chunks}...")
    try:
        df_raw = spark.read.format("xml").option("rowTag", "INDIVIDUAL").load(path_all_chunks)
    except Exception as e:
        print(f"Error reading XML chunks: {e}")
        return

    # --- Flattering: estrazione dei posto dalla gerarchia XML ---
    # Struttura INVIDUAL -> ID, WRITING (array) -> TITLE, DATE, INFO, TEXT
    df_posts = df_raw.select(
        col("ID").alias("subject_id"),
        explode(col("WRITING")).alias("post")
    ).select(
        col("subject_id"),
        trim(col("post.TITLE")).alias("title"),  #rimozione spazi vuoti
        col("post.DATE").cast("timestamp").alias("date"),  #casting da stringa a date
        col("post.INFO").alias("source_info"),
        trim(col("post.TEXT")).alias("text")
    )

    # ---Caricamento Label (TXT) ---
    print(f"Reading labels from: {path_labels}...")
    df_labels = spark.read.text(path_labels)

    # --- Mettiamo in ordine il formato (parsing) ---
    # Formato: "subject1234 1"
    df_labels_clean = df_labels.select(
        split(col("value"), "\s+").getItem(0).alias("subject_id"),
        split(col("value"), "\s+").getItem(1).cast(IntegerType()).alias("label")
    )

    # --- Join dei dati e delle label ---
    print(f"Execution JOIN between Posts and Labels...")
    #Inner join per tenere solo i post degli utenti etichettati
    df_full = df_posts.join(df_labels_clean, on="subject_id", how="inner")

    # --- Pipeline NLP per pulizia testo ---
    print(f"Starting pipeline NLP for cleaning and tokenization...")

    #1. Togliamo url/simboli e lasciamo tutto in lowercase
    df_cleaned = df_full.withColumn(
        "clean_text",
        lower(col("text"))
    ).withColumn(
        "clean_text",
        regexp_replace(col("clean_text"), r"http\S+", "")
    ).withColumn(
        "clean_text",
        regexp_replace(col("clean_text"), r"[^a-zA-Z\s]", "")
    )

    #2. Trasformazione da frase a lista di parole
    tokenizer = RegexTokenizer(inputCol="clean_text", outputCol="words", pattern="\\W")
    df_tokenized = tokenizer.transform(df_cleaned)

    #3. Rimozione delle parole irrilevanti
    remover = StopWordsRemover(inputCol="words", outputCol="filtered_words")
    df_final = remover.transform(df_tokenized)

    # --- Output e stats ---
    print("\n--- GLOBAL ANALYSIS RESULTS ---")

    # Count totale dei post
    print("Calculating statistics...")
    total_posts = df_final.count()
    depressed_posts = df_final.filter("label = 1").count()
    control_posts = total_posts - depressed_posts

    print(f"TOTAL POST PROCESSED: {total_posts}")
    print(f"Posts group 'Depression' (Label 1): {depressed_posts}")
    print(f"Posts group 'Control' (Label 0): {control_posts}")

    print("\n Processed Data Preview (Subject + Keywords):")
    df_final.select("subject_id", "label", "filtered_words").show(5, truncate=80)


if __name__ == "__main__":
    main()

