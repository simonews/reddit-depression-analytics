import os
import sys
from pyspark.sql import SparkSession

def get_spark_session(app_name='RedditAnalysis'):
    """
    creates a spark session configurated for my Windows environment and for read XML files
    """

    # ---1. Configurazione ambiente ---
    os.environ['HADOOP_HOME'] = r"x"
    os.environ['JAVA_HOME'] = r"x"

    sys_path = os.environ.get('PATH', '')
    if os.environ['HADOOP_HOME'] not in sys_path:
        os.environ['PATH'] = os.environ['HADOOP_HOME'] + "\\bin;" + \
                             os.environ['JAVA_HOME'] + "\\bin;" + \
                             sys_path

    print(f"Enviromnment configuration completed. JAVA_HOME: {os.environ['JAVA_HOME']}")

    try:
        import findspark
        findspark.init()
    except ImportError:
        pass


    # ---2. Creazione sessione spark ---
    try:
        #libreria esterna per leettura file xml
        xml_package = "com.databricks:spark-xml_2.12:0.17.0"
        warehouse_dir = os.getenv('SPARK_WAREHOUSE', "file:///C:/temp")

        print(f"Starting spark session with xml package: {xml_package}...")

        spark = SparkSession.builder.appName(app_name).master("local[*]") \
                                                      .config("spark.driver.memory", "4g") \
                                                      .config("spark.sql.shuffle.partitions", "200") \
                                                      .config("spark.sql.warehouse.dir", "file:///C:/temp") \
                                                      .config("spark.jars.packages", xml_package) \
                                                      .getOrCreate()

        spark.sparkContext.setLogLevel("WARN")
        print(f"Spark session configurated - version: {spark.version}")
        return spark

    except Exception as e:
        print(f"Critical error during the start: {e}")
        sys.exit(1)