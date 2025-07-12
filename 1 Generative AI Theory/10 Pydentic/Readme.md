## Introduction to Pydantic
Pydantic is a Python library that uses type annotations for runtime data validation, parsing, and serialization. It enforces data integrity by checking inputs against defined models, making it ideal for APIs, configurations, and data pipelines.

- **Core Theory**: Pydantic builds on Python's `typing` module and uses a metaclass to generate validation logic at class definition time. It performs type coercion (e.g., converting "42" to int 42) where safe, and raises `ValidationError` for failures. In v2+, it uses Rust for speed, reducing overhead in large-scale apps.
- **Key Features**: Automatic error messages, custom validators, JSON schema generation.
- **Installation**: `pip install pydantic` (Python 3.8+).
- **Resources**: Official docs at [pydantic.dev](https://docs.pydantic.dev/latest/).

## Practical Implementation Examples with Theory
Each example includes a "Theory" section explaining the underlying principles, followed by the code.

### Example 1: Basic Model Definition
#### Theory
In Pydantic, models are subclasses of `BaseModel`, where attributes use type hints for validation. When you call `model_validate()`, Pydantic parses the input (e.g., dict) and checks each field: it coerces types if possible (e.g., str to int), applies defaults, and validates built-in types like `EmailStr` (which uses regex for email format). If validation fails, it collects errors in a `ValidationError` exception, providing detailed messages. This promotes type safety without manual checks, reducing bugs in dynamic data handling.

```python
from pydantic import BaseModel, EmailStr

class User(BaseModel):
    id: int
    name: str
    email: EmailStr
    age: int = 18  # Default value

# Valid data example
user_data = {"id": 1, "name": "Alice", "email": "alice@example.com"}
user = User.model_validate(user_data)
print(user.model_dump())  # Output: {'id': 1, 'name': 'Alice', 'email': 'alice@example.com', 'age': 18}
```

### Example 2: Fields with Constraints
#### Theory
The `Field` class allows metadata like constraints (e.g., `min_length`, `gt` for greater than) and descriptions. These are enforced during validation using Pydantic's core engine, which integrates with `pydantic_core` (Rust-based in v2) for efficient checks. Regex patterns (`pattern`) use Python's `re` module under the hood. This theory draws from schema-driven validation, similar to JSON Schema, ensuring data conforms to business rules before processing.

```python
from pydantic import BaseModel, Field

class Product(BaseModel):
    name: str = Field(min_length=3, max_length=50)
    price: float = Field(gt=0, description="Price must be positive")
    sku: str = Field(pattern=r'^\d{3}-\w{4}$')  # Regex example

# Valid product
product = Product(name="Widget", price=9.99, sku="123-ABCD")
print(product.model_dump_json())  # JSON output
```

### Example 3: Custom Validators
#### Theory
Validators are functions decorated with `@field_validator` or `@model_validator` that run after initial type checks. They allow custom logic, raising `ValueError` for failures, which Pydantic wraps in `ValidationError`. This extends Pydantic's declarative approach with imperative code, enabling complex validations like password strength or cross-field checks. The classmethod decorator ensures access to the class context, and validators can be reusable across models.

```python
from pydantic import BaseModel, field_validator

class Account(BaseModel):
    username: str
    password: str

    @field_validator('password')
    @classmethod
    def validate_password(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError('Password must be at least 8 characters')
        return v

# Valid account
account = Account(username="user1", password="securepass")
print(account.model_dump())
```

### Example 4: Nested Models and Lists
#### Theory
Pydantic supports composition: models can nest other models or use typing constructs like `List[T]`. During validation, it recursively parses nested data, applying the same coercion and checks. This follows object-oriented principles for hierarchical data, with lazy evaluation to handle large structures efficiently. Errors propagate with paths (e.g., "employees[0].email") for precise debugging.

```python
from pydantic import BaseModel
from typing import List

class Address(BaseModel):
    street: str
    city: str

class Company(BaseModel):
    name: str
    employees: List[User]  # Reuses User from Example 1
    headquarters: Address

# Sample data
data = {
    "name": "TechCorp",
    "employees": [{"id": 1, "name": "Alice", "email": "alice@techcorp.com"}],
    "headquarters": {"street": "123 Main St", "city": "Anytown"}
}
company = Company.model_validate(data)
print(company.headquarters.city)  # Output: Anytown
```

### Example 5: Serialization and Excludes
#### Theory
Serialization methods like `model_dump()` convert models to dicts, optionally excluding fields or using aliases. `model_json_schema()` generates a JSON Schema dict, useful for API docs or frontend validation. This is based on Pydantic's internal schema builder, which reflects the model's type hints and fields into a standard format. Excludes help with data privacy (e.g., hiding sensitive fields), and the process is optimized for performance in high-throughput scenarios.

```python
# Assuming User from Example 1
user = User(id=1, name="Alice", email="alice@example.com")
print(user.model_dump(exclude={"email"}))  # Excludes email field
print(user.model_json_schema())  # Generates JSON Schema
```

## Advanced Theory and Tips
- **Overall Validation Flow**: Pydantic's pipeline: Parse input → Coerce types → Run field validators → Run model validators → Return instance or raise error.
- **Performance Considerations**: v2+ uses compiled Rust validators, making it 10-50x faster than v1 for large datasets.
- **Error Handling Best Practices**: Always wrap in try-except to catch `ValidationError` and access `e.errors()` for JSON-friendly error lists.
- **Integration Theory**: In FastAPI, Pydantic models auto-validate request bodies, leveraging OpenAPI schemas for swagger docs.

If you need more theory (e.g., on settings management or unions), additional examples, or updates from latest docs, let me know!
