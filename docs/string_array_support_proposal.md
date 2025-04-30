# Implementation Proposal: Support for Array<String> in caspi.torch

This document outlines the changes needed to support arrays of strings in the `caspi.torch` module of the caspi library.

## Current Limitations

Currently, the library has the following limitations:

1. The schema validation in `_validate_df_schema` rejects arrays of strings
2. The `_arrow_batch_to_tensor_dict` function doesn't have logic to handle arrays of strings

## Proposed Implementation

### 1. Update Schema Validation

First, update the `_validate_df_schema` function in `helpers.py` to allow arrays of strings if a tokenizer is provided:

```python
def _validate_df_schema(
    df: DataFrame, timestamp_to: str | None = None, tokenizer: Callable | None = None
) -> None:
    # ...
    
    allowed_array_element_types: tuple[type, ...] = (NumericType, BooleanType)
    if timestamp_to == "int64":  # Be specific about supported conversion
        allowed_array_element_types += (TimestampType,)
    
    for field in df.schema.fields:
        dt = field.dataType
        # ... (existing checks)
        
        if isinstance(dt, ArrayType):
            element_type = dt.elementType
            if isinstance(element_type, allowed_array_element_types):
                continue
                
            # Allow array of strings if tokenizer is provided
            if isinstance(element_type, StringType) and tokenizer is not None:
                continue

            # Example: Allow Array<Array<Numeric>>
            # if isinstance(element_type, ArrayType) and \
            #    isinstance(element_type.elementType, allowed_array_element_types):
            #      continue

        # ... (existing error raising code)
```

### 2. Update Tensor Conversion

Next, update the `_arrow_batch_to_tensor_dict` function to handle arrays of strings by tokenizing each array individually:

```python
def _arrow_batch_to_tensor_dict(
    record_batch: pa.RecordBatch,
    spark_schema: StructType,
    tokenizer: Callable | None = None,
) -> TensorDict:
    # ...
    
    for i, field in enumerate(spark_schema.fields):
        col_name = field.name
        col_type = field.dataType
        arrow_array = record_batch.column(i)
        
        # ... (existing code for other types)
        
        elif isinstance(col_type, ArrayType):
            processed_list: list[TensorBatch] = []
            inner_type = col_type.elementType
            
            if isinstance(inner_type, NumericType):
                empty_dtype = torch.float32
            elif isinstance(inner_type, BooleanType):
                empty_dtype = torch.bool
            elif isinstance(inner_type, TimestampType):
                empty_dtype = torch.int64
            elif isinstance(inner_type, StringType):
                # For string arrays, we need special handling
                assert tokenizer is not None, (
                    "Tokenizer must be provided for StringType arrays."
                )
                
                # Process each array separately
                for item in arrow_array.to_pylist():
                    if item is None:
                        # For null arrays, add an empty tensor or empty dict
                        processed_list.append(torch.empty(0, dtype=torch.float32))
                    else:
                        # Convert None to empty string and apply tokenizer
                        item_texts = ["" if x is None else str(x) for x in item]
                        encodings = tokenizer(
                            item_texts,
                            padding="longest",
                            truncation=True,
                            return_tensors="pt",
                        )
                        processed_list.append(encodings)
                
                tensors[col_name] = processed_list
                continue  # Skip the rest of the processing for this column
            else:
                empty_dtype = torch.float32
                
            # ... (existing code for non-string arrays)
```

### 3. Update Concatenation Logic

The `_concatenate_tensor_dicts` function might need updates to handle string array tensors correctly, especially if the tokenizer returns dictionaries:

```python
def _concatenate_tensor_dicts(dict1: TensorDict, dict2: TensorDict) -> TensorDict:
    # ...
    
    for key in dict1:
        val1 = dict1[key]
        val2 = dict2[key]
        
        if isinstance(val1, torch.Tensor) and isinstance(val2, torch.Tensor):
            # ... (existing tensor concatenation code)
        elif isinstance(val1, list) and isinstance(val2, list):
            # Check if we have lists of dictionary items (tokenized string arrays)
            if val1 and isinstance(val1[0], dict) and val2 and isinstance(val2[0], dict):
                # For tokenizer outputs (dictionaries), combine the lists
                concatenated[key] = val1 + val2
            else:
                # ... (existing list concatenation code)
        else:
            # ... (existing error code)
```

## Testing

The tests in `test_string_array_loader.py` could be updated to verify the implementation works as expected after these changes are made.

## Conclusion

These changes would enable the library to handle arrays of strings, which would be tokenized into appropriate tensor representations, making the library more versatile for NLP and text processing tasks.
