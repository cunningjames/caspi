"""Tests for loading DataFrames with string array columns."""

import pyarrow as pa
import torch
from pyspark.sql import SparkSession
from pyspark.sql.types import (
    ArrayType,
    StringType,
    StructField,
    StructType,
)

from caspi.torch import loader
from caspi.torch.helpers import _arrow_batch_to_tensor_dict

# --- Helper Functions/Classes ---

class DummyTokenizer:
    """Minimal tokenizer for testing StringType conversion."""

    def __call__(
        self,
        texts: list[str],
        padding: str = "longest",
        truncation: bool = True,
        return_tensors: str = "pt",
    ) -> dict[str, torch.Tensor]:
        """Tokenizes a list of strings into tensors.
        
        This is a simple mock implementation that:
        1. Replaces None/empty strings with empty sequence
        2. Creates a single token (0) for each non-empty string
        3. Creates appropriate attention mask (1 for tokens, 0 for padding)
        
        Args:
            texts (list[str]): List of input strings to tokenize
            padding (str): Padding strategy (ignored in this mock)
            truncation (bool): Whether to truncate (ignored in this mock)
            return_tensors (str): Output format (ignored in this mock)
            
        Returns:
            dict[str, torch.Tensor]: Dictionary with "input_ids" and 
                "attention_mask" tensors
        """
        # Create a simple encoding: token 0 for each non-empty string
        # with appropriate attention mask
        input_ids = []
        attention_mask = []
        
        for text in texts:
            if text and text.strip():
                input_ids.append([0])  # Just use token 0 for any text
                attention_mask.append([1])
            else:
                input_ids.append([0])  # Use same token but no attention
                attention_mask.append([0])
                
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.int64),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.int64),
        }


# --- Test Functions ---

def test_arrow_batch_to_tensor_dict_string_array() -> None:
    """Tests _arrow_batch_to_tensor_dict with arrays of strings.
    
    This test verifies that arrays of strings are properly converted to
    tokenized tensors when a tokenizer is provided.
    """
    # Create schema with string array column
    schema = StructType([
        StructField("string_array_col", ArrayType(StringType(), True), True),
    ])
    
    # Create array with string arrays, including nulls at various levels
    string_arrays = [
        ["hello", "world", None],  # Array with null element
        None,                       # Null array
        ["goodbye"],                # Simple array
        []                          # Empty array
    ]
    
    array_data = pa.array(string_arrays, type=pa.list_(pa.string()))
    
    rb = pa.RecordBatch.from_arrays(
        [array_data], 
        schema=pa.schema([pa.field("string_array_col", pa.list_(pa.string()), nullable=True)])
    )
    
    tokenizer = DummyTokenizer()
    
    # With our implementation, this should now work
    tensor_dict = _arrow_batch_to_tensor_dict(rb, schema, tokenizer=tokenizer)
    
    assert "string_array_col" in tensor_dict
    assert isinstance(tensor_dict["string_array_col"], list)
    assert len(tensor_dict["string_array_col"]) == 4
    
    # First item: array with values and null
    assert isinstance(tensor_dict["string_array_col"][0], dict)
    assert "input_ids" in tensor_dict["string_array_col"][0]
    assert "attention_mask" in tensor_dict["string_array_col"][0]
    assert tensor_dict["string_array_col"][0]["input_ids"].shape[0] == 3
    
    # Second item: null array becomes empty tensor
    assert isinstance(tensor_dict["string_array_col"][1], torch.Tensor)
    assert tensor_dict["string_array_col"][1].numel() == 0
    
    # Third item: simple array
    assert isinstance(tensor_dict["string_array_col"][2], dict)
    assert tensor_dict["string_array_col"][2]["input_ids"].shape[0] == 1
    
    # Fourth item: empty array also becomes a dictionary with empty tensors
    assert isinstance(tensor_dict["string_array_col"][3], dict)
    assert "input_ids" in tensor_dict["string_array_col"][3]
    assert "attention_mask" in tensor_dict["string_array_col"][3]
    assert tensor_dict["string_array_col"][3]["input_ids"].numel() == 0


def test_loader_with_string_array(spark: SparkSession) -> None:
    """Tests creating a loader for DataFrame with string array column.
    
    This test verifies that a DataFrame with a column containing arrays of strings
    can be properly loaded and processed when using a tokenizer.
    """
    # Define schema with string array column
    from pyspark.sql.types import IntegerType
    
    schema = StructType([
        StructField("id", IntegerType(), False),
        StructField("texts", ArrayType(StringType(), True), True),
    ])
    
    # Create sample data
    data = [
        (1, ["hello", "world"]),
        (2, ["test"]),
        (3, None),
        (4, []),
        (5, ["multiple", "strings", "in", "array"]),
    ]
    
    # Create DataFrame
    df = spark.createDataFrame(data, schema=schema)
    
    # Create tokenizer
    tokenizer = DummyTokenizer()
    
    # With our implementation, this should now work
    data_loader = loader(df, batch_size=2, tokenizer=tokenizer)
    
    # Collect and verify batches
    batches = list(data_loader)
    assert len(batches) == 3  # 5 rows with batch_size=2 should give 3 batches
    
    # Check that the id column is properly processed
    ids = []
    for batch in batches:
        ids.extend(batch["id"].tolist())
    assert ids == [1, 2, 3, 4, 5]
    
    # Check that the texts are properly processed
    for batch in batches:
        assert "texts" in batch
        assert isinstance(batch["texts"], list)
        for item in batch["texts"]:
            if isinstance(item, dict):
                assert "input_ids" in item
                assert "attention_mask" in item
            else:
                # Should be an empty tensor for None or [] values
                assert isinstance(item, torch.Tensor) and item.numel() == 0
