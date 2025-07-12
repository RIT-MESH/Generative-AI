# Pydantic Tutorial

Welcome to this document, It will render with headings, lists, and syntax-highlighted code blocks for easy reading and execution. If you need modifications or more sections, let me know!

## Introduction to Pydantic
Pydantic is a Python library for data validation, parsing, and serialization using type hints. It's fast, flexible, and integrates well with tools like FastAPI.

- **Key Features**:
  - Automatic validation based on type annotations.
  - Support for constraints (e.g., min/max values, regex).
  - Custom validators for complex logic.
  - Nested models, lists, and optional fields.
  - JSON serialization and schema generation.

- **Installation**: `pip install pydantic` (requires Python 3.8+).
- **Resources**: Official docs at [pydantic.dev](https://docs.pydantic.dev/latest/).

## Practical Implementation Examples
Each example is a standalone code block. Copy and run them in your Python environment.

### Example 1: Basic Model Definition
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
```python
# Assuming User from Example 1
user = User(id=1, name="Alice", email="alice@example.com")
print(user.model_dump(exclude={"email"}))  # Excludes email field
print(user.model_json_schema())  # Generates JSON Schema
```

## Advanced Tips
- For errors: Catch `ValidationError` and print `e.errors()` for details.
- Integration: Use with FastAPI for request/response models.
- Experiment: Try invalid data in the examples to see validation in action.

If this "window" needs expansion (e.g., more examples or diagrams), reply with details!
