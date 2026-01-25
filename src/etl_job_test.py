from utils import get_spark_session
from pyspark.sql.functions import explode, col, trim

def main():
    #========================
    # STARTING SPARK SESSION
    #========================
    spark = get_spark_session("ETL_read_test")

    #=========================
    # DEFINITION FILE TO READ
    #=========================
    path_chunk1 = "data/chunk1/*.xml"

    print(f"Attempting to read file from: {path_chunk1}")

    try:
        #============
        # XML READING
        #============
        df_raw = spark.read \
                 .format("xml") \
                 .option("rowTag", "INDIVIDUAL") \
                 .load(path_chunk1)

        print("Rough pattern detected:")
        df_raw.printSchema()

        #============
        # FLATTERING
        #============
        # Structure: INVIDUAL -> ID, WRITING (array) -> TITLE, DATE, INFO, TEXT
        df_posts = df_raw \
            .select(
            col("ID").alias("subject_id"),
            explode(col("WRITING")).alias("post")
            ) \
                .select(
                col("subject_id"),
                trim(col("post.TITLE")).alias("title"),  # Empty space remove
                col("post.DATE").cast("timestamp").alias("date"),  # String-data convertion
                col("post.INFO").alias("source_info"),
                trim(col("post.TEXT")).alias("text")
            )

        #===============
        # VERIFY OUTPUT
        #===============
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
