from utils import get_spark_session
from pyspark.sql.functions import col,explode, lower, trim, split, length, desc
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import os

def generate_wordcloud(word_freqs, title, filename):
    #===========================================
    # GENERATE AND SAVE A WORDCLOUD FROM A DICT
    #===========================================
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(word_freqs)

    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis("off")
    plt.title(title, fontsize=15)
    plt.savefig(filename)
    print(f"Wordcloud saved to {filename}")
    plt.close()

def main():
    spark = get_spark_session()

    #==================
    # FAST DATA RELOAD
    #==================
    print(f"Data reloading and preparing for the visualization...")

    path_all_chunks = "data/chunk*/*.xml"
    path_labels = "data/risk-golden-truth-test.txt"

    #===========
    # LOAD XML
    #===========
    df_raw = spark.read.format("xml").option("rowTag", "INDIVIDUAL").load(path_all_chunks)
    df_posts = df_raw.select(col("ID").alias("subject_id"), explode(col("WRITING")).alias("post")) \
        .select(col("subject_id"), lower(trim(col("post.TEXT"))).alias("text"))

    #============
    # LOAD LABELS
    #============
    df_labels = spark.read.text(path_labels)
    df_labels_clean = df_labels.select(
        split(col("value"), "\s+").getItem(0).alias("subject_id"),
        split(col("value"), "\s+").getItem(1).cast("int").alias("label")
    )

    #======
    # JOIN
    #======
    df_full = df_posts.join(df_labels_clean, on="subject_id", how="inner")

    #===============
    # WORD ANALYSIS
    #===============
    custom_stop_words = [
        # Articles and prepositions
        "the", "a", "an", "and", "or", "but", "if", "because", "as", "what",
        "when", "where", "how", "which", "who", "whom", "why", "from", "to",
        "of", "in", "on", "at", "by", "for", "with", "about", "into", "through",
        "during", "before", "after", "above", "below", "up", "down", "out", "off",
        "over", "under", "again", "further", "then", "once", "here", "there",

        # Verbs
        "is", "are", "was", "were", "be", "been", "being", "am", "im", "ive",
        "have", "has", "had", "having", "do", "does", "did", "doing",
        "can", "could", "shall", "should", "will", "would", "may", "might", "must",

        # Pronouns
        "i", "me", "my", "we", "our", "ours", "ourselves",
        "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself",
        "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
        "theirs", "themselves", "this", "that", "these", "those",

        # Empty words
        "just", "really", "very", "so", "like", "much", "many", "some", "any", "no", "not",
        "only", "own", "same", "too", "than", "also", "even", "back", "get", "go", "going", "got",
        "well", "way", "made", "make", "know", "think", "thing", "things", "see", "say", "said",
        "want", "good", "better", "best", "still", "never", "ever", "last",
        "right", "left", "sure", "actually", "maybe", "probably", "one", "two", "people",
        "someone", "something", "time", "years", "day", "days", "life", "look", "use",
        "all", "more", "now", "first", "since", "while", "lot", "little", "few", "take",

        # Tech residues and web
        "http", "https", "www", "com", "reddit", "subreddit", "post", "posts", "comment", "comments",
        "img", "png", "jpg", "html", "php", "url", "link", "deleted", "removed", "org",

        # Artifacts
        "don", "didn", "doesn", "isn", "wasn", "weren", "hasn", "haven", "hadn",
        "won", "wouldn", "couldn", "shouldn", "cant", "cannot", "wont", "arent"
    ]

    print("Calculating words frequency for group 'Depression' (Label=1)...")
    #Filter depressed -> break the words -> filter stop words -> count
    df_depressed = df_full.filter("label = 1") \
        .withColumn("word", explode(split(col("text"), "\\W+"))) \
        .filter(~col("word").isin(custom_stop_words)) \
        .filter(length(col("word")) > 2) \
        .groupBy("word").count().orderBy(desc("count")).limit(100)

    #===================================
    # CONVERT TO LOCAL DICT (Few words)
    #===================================
    freq_depressed = {row['word']: row['count'] for row in df_depressed.collect()}

    print(f"Calculating words frequency for group 'Control' (Label=0)...")
    df_control = df_full.filter("label = 0") \
        .withColumn("word", explode(split(col("text"), "\\W+"))) \
        .filter(~col("word").isin(custom_stop_words)) \
        .filter(length(col("word")) > 2) \
        .groupBy("word").count().orderBy(desc("count")).limit(100)

    freq_control = {row['word']: row['count'] for row in df_control.collect()}

    #===================
    # GRAPH GENERATION
    #===================
    if not os.path.exists("plots"):
        os.makedirs("plots")

    generate_wordcloud(freq_depressed, title="Top words - users at risk", filename="plots/wordcloud_depression1.png")
    generate_wordcloud(freq_control, title="Top words - users control", filename="plots/wordcloud_control1.png")

    print("\n Work done. Check the repository 'plots' for visualization.")

if __name__ == "__main__":
    main()