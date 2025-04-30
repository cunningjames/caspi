import pytest
from pyspark.sql import SparkSession
from collections.abc import Iterator


@pytest.fixture(scope="session")
def spark() -> Iterator[SparkSession]:
    """Creates a SparkSession for testing.

    Returns:
        Iterator[SparkSession]: An iterator yielding the SparkSession instance.
    """
    spark_session = (
        SparkSession.builder.master("local[1]")
        .appName("pytest-spark-session")
        .config("spark.sql.shuffle.partitions", "1")
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        .config("spark.sql.execution.arrow.maxRecordsPerBatch", "5") # Small batch size for testing
        .getOrCreate()
    )
    yield spark_session
    spark_session.stop()
