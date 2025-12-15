from utils import get_spark_session
from pyspark.sql.functions import explode, col, trim

def main():
    # ---1. Avvio di spark ---
    spark = get_spark_session("ETL_read_test")

    # ---2. Definizione del file da leggere ---
    path_chunk1 = "data/chunk1/*.xml"

    print(f"Attempting to read file from: {path_chunk1}")

    try:
        # ---3. Lettura XML ---
        df_raw = spark.read \
                 .format("xml") \
                 .option("rowTag", "INDIVIDUAL") \
                 .load(path_chunk1)

        print("Rough pattern detected:")
        df_raw.printSchema()

        # ---4. Flattering ---
        #Struttura INVIDUAL -> ID, WRITING (array) -> TITLE, DATE, INFO, TEXT
        df_posts = df_raw \
            .select(
            col("ID").alias("subject_id"),
            explode(col("WRITING")).alias("post")
            ) \
                .select(
                col("subject_id"),
                trim(col("post.TITLE")).alias("title"),  #Rimozione spazi vuoti
                col("post.DATE").cast("timestamp").alias("date"),  #Conversione stringa-data
                col("post.INFO").alias("source_info"),
                trim(col("post.TEXT")).alias("text")
            )

        # ---5. Output di verifica ---
        print("Final schema:")
        df_posts.printSchema()

        print("Data preview (first 5 records):")
        df_posts.show(5, truncate=40)

        total_posts = df_posts.count()
        print(f"Ingestion completed - total posts in chunk1: {total_posts}")

    except Exception as e:
        print(f"Critical error during the start: {e}")

if __name__ == "__main__":
    main()

