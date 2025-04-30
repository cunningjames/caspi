- Generate docstrings using the Google format.
- Function definitions should be annotated with argument and return types.
- Obey the following rules when adding type annotations:
  - Use the lower-case version of builtin collection types: `list` rather than `List`, `dict` rather than `Dict`, and so on.
  - Use the syntax `SomeType | None` rather than `Optional[SomeType]`.
  - Use the syntax `TypeA | TypeB` rather than `Union[TypeA, TypeB]`.
- Never add comments using the `#` syntax. When necessary, difficult logic should be explained in the relevant docstring.

Python example:

```python
def factorial(n: int, acc: list[int] | None = None) -> list[int]:
    """Computes the factorial of a number tail-recursively.
    
    Args:
        n (int): the number whose factorial will be computed
        acc (list[int] | None): the accumulated factorial values
        
    Returns:
        the accumulated list of multiplied values, with the
            final value being the factorial of n
        
    Raises:
        TypeError: if n or acc are not of the correct types
        ValueError: if n < 0 or acc < 0
    """
    
    if not isinstance(n, int) or not isinstance(acc, list | None):
        raise TypeError("arguments have incorrect types")
    if n < 0 or (acc is not None and acc[-1] < 0):
        raise ValueError("arguments must be non-negative")
    if acc is None:
        acc = [1]
    if n in [0, 1]:
        return acc

    return factorial(n - 1, acc + [acc[-1] * n])
```
