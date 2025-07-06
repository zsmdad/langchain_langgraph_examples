# 大量工具调用的例子，大模型直接决定用什么工具
import os
import json
import re
import datetime
import random
import math
import hashlib
import base64
import uuid
from typing import Annotated, TypedDict, List, Dict, Optional, Union, Any

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

load_dotenv()

class State(TypedDict):
    messages: Annotated[list, add_messages]

@tool
def add_numbers(a: float, b: float) -> float:
    """Add two numbers together."""
    return a + b

@tool
def subtract_numbers(a: float, b: float) -> float:
    """Subtract second number from first number."""
    return a - b

@tool
def multiply_numbers(a: float, b: float) -> float:
    """Multiply two numbers together."""
    return a * b

@tool
def divide_numbers(a: float, b: float) -> float:
    """Divide first number by second number."""
    if b == 0:
        raise ValueError("Cannot divide by zero")
    return a / b

@tool
def calculate_power(base: float, exponent: float) -> float:
    """Calculate base raised to the power of exponent."""
    return base ** exponent

@tool
def calculate_square_root(number: float) -> float:
    """Calculate square root of a number."""
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number")
    return math.sqrt(number)

@tool
def calculate_factorial(n: int) -> int:
    """Calculate factorial of a non-negative integer."""
    if n < 0:
        raise ValueError("Factorial is not defined for negative numbers")
    return math.factorial(n)

@tool
def calculate_gcd(a: int, b: int) -> int:
    """Calculate greatest common divisor of two integers."""
    return math.gcd(a, b)

@tool
def calculate_lcm(a: int, b: int) -> int:
    """Calculate least common multiple of two integers."""
    return abs(a * b) // math.gcd(a, b)

@tool
def is_prime(number: int) -> bool:
    """Check if a number is prime."""
    if number < 2:
        return False
    for i in range(2, int(math.sqrt(number)) + 1):
        if number % i == 0:
            return False
    return True

# 字符串处理工具
@tool
def string_length(text: str) -> int:
    """Get the length of a string."""
    return len(text)

@tool
def string_upper(text: str) -> str:
    """Convert string to uppercase."""
    return text.upper()

@tool
def string_lower(text: str) -> str:
    """Convert string to lowercase."""
    return text.lower()

@tool
def string_reverse(text: str) -> str:
    """Reverse a string."""
    return text[::-1]

@tool
def string_replace(text: str, old: str, new: str) -> str:
    """Replace occurrences of old substring with new substring."""
    return text.replace(old, new)

@tool
def string_split(text: str, delimiter: str = " ") -> List[str]:
    """Split string by delimiter."""
    return text.split(delimiter)

@tool
def string_join(strings: List[str], separator: str = " ") -> str:
    """Join list of strings with separator."""
    return separator.join(strings)

@tool
def string_strip(text: str) -> str:
    """Remove leading and trailing whitespace."""
    return text.strip()

@tool
def count_words(text: str) -> int:
    """Count number of words in text."""
    return len(text.split())

@tool
def find_substring(text: str, substring: str) -> int:
    """Find first occurrence of substring in text. Returns -1 if not found."""
    return text.find(substring)

# 日期时间工具
@tool
def get_current_time() -> str:
    """Get current date and time."""
    return datetime.datetime.now().isoformat()

@tool
def get_current_date() -> str:
    """Get current date."""
    return datetime.date.today().isoformat()

@tool
def add_days_to_date(date_str: str, days: int) -> str:
    """Add specified number of days to a date."""
    date_obj = datetime.datetime.fromisoformat(date_str)
    new_date = date_obj + datetime.timedelta(days=days)
    return new_date.isoformat()

@tool
def calculate_date_difference(date1: str, date2: str) -> int:
    """Calculate difference in days between two dates."""
    d1 = datetime.datetime.fromisoformat(date1)
    d2 = datetime.datetime.fromisoformat(date2)
    return (d2 - d1).days

@tool
def format_date(date_str: str, format_string: str) -> str:
    """Format date according to format string."""
    date_obj = datetime.datetime.fromisoformat(date_str)
    return date_obj.strftime(format_string)

@tool
def get_weekday(date_str: str) -> str:
    """Get weekday name for a given date."""
    date_obj = datetime.datetime.fromisoformat(date_str)
    return date_obj.strftime("%A")

@tool
def is_weekend(date_str: str) -> bool:
    """Check if a date falls on weekend."""
    date_obj = datetime.datetime.fromisoformat(date_str)
    return date_obj.weekday() >= 5

# 数据结构操作工具
@tool
def list_length(items: List[Any]) -> int:
    """Get length of a list."""
    return len(items)

@tool
def list_sum(numbers: List[float]) -> float:
    """Calculate sum of numbers in a list."""
    return sum(numbers)

@tool
def list_average(numbers: List[float]) -> float:
    """Calculate average of numbers in a list."""
    if not numbers:
        raise ValueError("Cannot calculate average of empty list")
    return sum(numbers) / len(numbers)

@tool
def list_max(numbers: List[float]) -> float:
    """Find maximum value in a list."""
    if not numbers:
        raise ValueError("Cannot find max of empty list")
    return max(numbers)

@tool
def list_min(numbers: List[float]) -> float:
    """Find minimum value in a list."""
    if not numbers:
        raise ValueError("Cannot find min of empty list")
    return min(numbers)

@tool
def list_sort(items: List[Any], reverse: bool = False) -> List[Any]:
    """Sort a list in ascending or descending order."""
    return sorted(items, reverse=reverse)

@tool
def list_unique(items: List[Any]) -> List[Any]:
    """Remove duplicates from a list while preserving order."""
    seen = set()
    result = []
    for item in items:
        if item not in seen:
            seen.add(item)
            result.append(item)
    return result

@tool
def list_filter_positive(numbers: List[float]) -> List[float]:
    """Filter positive numbers from a list."""
    return [n for n in numbers if n > 0]

@tool
def list_filter_even(numbers: List[int]) -> List[int]:
    """Filter even numbers from a list."""
    return [n for n in numbers if n % 2 == 0]

# 编码/解码工具
@tool
def base64_encode(text: str) -> str:
    """Encode text to base64."""
    return base64.b64encode(text.encode()).decode()

@tool
def base64_decode(encoded: str) -> str:
    """Decode base64 text."""
    return base64.b64decode(encoded).decode()

@tool
def url_encode(text: str) -> str:
    """URL encode text."""
    import urllib.parse
    return urllib.parse.quote(text)

@tool
def url_decode(encoded: str) -> str:
    """URL decode text."""
    import urllib.parse
    return urllib.parse.unquote(encoded)

@tool
def md5_hash(text: str) -> str:
    """Generate MD5 hash of text."""
    return hashlib.md5(text.encode()).hexdigest()

@tool
def sha256_hash(text: str) -> str:
    """Generate SHA256 hash of text."""
    return hashlib.sha256(text.encode()).hexdigest()

# JSON处理工具
@tool
def json_parse(json_string: str) -> Dict[str, Any]:
    """Parse JSON string to dictionary."""
    return json.loads(json_string)

@tool
def json_stringify(data: Dict[str, Any], indent: int = 2) -> str:
    """Convert dictionary to JSON string."""
    return json.dumps(data, indent=indent)

@tool
def json_get_keys(json_string: str) -> List[str]:
    """Get all keys from JSON object."""
    data = json.loads(json_string)
    return list(data.keys()) if isinstance(data, dict) else []

@tool
def json_get_value(json_string: str, key: str) -> Any:
    """Get value for specific key from JSON object."""
    data = json.loads(json_string)
    return data.get(key) if isinstance(data, dict) else None

# 随机数生成工具
@tool
def random_integer(min_val: int, max_val: int) -> int:
    """Generate random integer between min and max (inclusive)."""
    return random.randint(min_val, max_val)

@tool
def random_float(min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Generate random float between min and max."""
    return random.uniform(min_val, max_val)

@tool
def random_choice(items: List[Any]) -> Any:
    """Choose random item from a list."""
    if not items:
        raise ValueError("Cannot choose from empty list")
    return random.choice(items)

@tool
def random_shuffle(items: List[Any]) -> List[Any]:
    """Shuffle a list randomly."""
    shuffled = items.copy()
    random.shuffle(shuffled)
    return shuffled

@tool
def generate_uuid() -> str:
    """Generate a random UUID."""
    return str(uuid.uuid4())

# 文本分析工具
@tool
def count_characters(text: str) -> int:
    """Count total characters in text."""
    return len(text)

@tool
def count_lines(text: str) -> int:
    """Count number of lines in text."""
    return len(text.splitlines())

@tool
def count_vowels(text: str) -> int:
    """Count vowels in text."""
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char in vowels)

@tool
def count_consonants(text: str) -> int:
    """Count consonants in text."""
    vowels = "aeiouAEIOU"
    return sum(1 for char in text if char.isalpha() and char not in vowels)

@tool
def extract_emails(text: str) -> List[str]:
    """Extract email addresses from text."""
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    return re.findall(email_pattern, text)

@tool
def extract_urls(text: str) -> List[str]:
    """Extract URLs from text."""
    url_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.findall(url_pattern, text)

@tool
def extract_phone_numbers(text: str) -> List[str]:
    """Extract phone numbers from text."""
    phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
    return re.findall(phone_pattern, text)

# 数据验证工具
@tool
def is_email_valid(email: str) -> bool:
    """Validate email address format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))

@tool
def is_url_valid(url: str) -> bool:
    """Validate URL format."""
    pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w)*)?)?$'
    return bool(re.match(pattern, url))

@tool
def is_phone_valid(phone: str) -> bool:
    """Validate phone number format."""
    pattern = r'^\d{3}[-.]?\d{3}[-.]?\d{4}$'
    return bool(re.match(pattern, phone))

@tool
def is_integer(value: str) -> bool:
    """Check if string represents an integer."""
    try:
        int(value)
        return True
    except ValueError:
        return False

@tool
def is_float(value: str) -> bool:
    """Check if string represents a float."""
    try:
        float(value)
        return True
    except ValueError:
        return False

# 单位转换工具
@tool
def celsius_to_fahrenheit(celsius: float) -> float:
    """Convert Celsius to Fahrenheit."""
    return (celsius * 9/5) + 32

@tool
def fahrenheit_to_celsius(fahrenheit: float) -> float:
    """Convert Fahrenheit to Celsius."""
    return (fahrenheit - 32) * 5/9

@tool
def meters_to_feet(meters: float) -> float:
    """Convert meters to feet."""
    return meters * 3.28084

@tool
def feet_to_meters(feet: float) -> float:
    """Convert feet to meters."""
    return feet / 3.28084

@tool
def kg_to_pounds(kg: float) -> float:
    """Convert kilograms to pounds."""
    return kg * 2.20462

@tool
def pounds_to_kg(pounds: float) -> float:
    """Convert pounds to kilograms."""
    return pounds / 2.20462

# 文件路径操作工具
@tool
def get_file_extension(filename: str) -> str:
    """Get file extension from filename."""
    import os
    return os.path.splitext(filename)[1]

@tool
def get_file_name(filepath: str) -> str:
    """Get filename from file path."""
    import os
    return os.path.basename(filepath)

@tool
def get_directory_name(filepath: str) -> str:
    """Get directory name from file path."""
    import os
    return os.path.dirname(filepath)

@tool
def join_paths(path1: str, path2: str) -> str:
    """Join two paths together."""
    import os
    return os.path.join(path1, path2)

# 数据格式化工具
@tool
def format_number(number: float, decimal_places: int = 2) -> str:
    """Format number with specified decimal places."""
    return f"{number:.{decimal_places}f}"

@tool
def format_currency(amount: float, currency: str = "USD") -> str:
    """Format amount as currency."""
    return f"{currency} {amount:.2f}"

@tool
def format_percentage(value: float, decimal_places: int = 1) -> str:
    """Format value as percentage."""
    return f"{value:.{decimal_places}f}%"

# 集合操作工具
@tool
def set_union(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Get union of two lists (unique elements from both)."""
    return list(set(list1) | set(list2))

@tool
def set_intersection(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Get intersection of two lists (common elements)."""
    return list(set(list1) & set(list2))

@tool
def set_difference(list1: List[Any], list2: List[Any]) -> List[Any]:
    """Get difference of two lists (elements in first but not second)."""
    return list(set(list1) - set(list2))

# 高级计算工具
@tool
def calculate_distance_2d(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate Euclidean distance between two 2D points."""
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2)

@tool
def calculate_circle_area(radius: float) -> float:
    """Calculate area of a circle."""
    return math.pi * radius**2

@tool
def calculate_rectangle_area(width: float, height: float) -> float:
    """Calculate area of a rectangle."""
    return width * height

@tool
def calculate_triangle_area(base: float, height: float) -> float:
    """Calculate area of a triangle."""
    return 0.5 * base * height

# 示例工具列表
tools = [
    add_numbers, subtract_numbers, multiply_numbers, divide_numbers,
    calculate_power, calculate_square_root, calculate_factorial,
    calculate_gcd, calculate_lcm, is_prime,
    string_length, string_upper, string_lower, string_reverse,
    string_replace, string_split, string_join, string_strip,
    count_words, find_substring,
    get_current_time, get_current_date, add_days_to_date,
    calculate_date_difference, format_date, get_weekday, is_weekend,
    list_length, list_sum, list_average, list_max, list_min,
    list_sort, list_unique, list_filter_positive, list_filter_even,
    base64_encode, base64_decode, url_encode, url_decode,
    md5_hash, sha256_hash,
    json_parse, json_stringify, json_get_keys, json_get_value,
    random_integer, random_float, random_choice, random_shuffle, generate_uuid,
    count_characters, count_lines, count_vowels, count_consonants,
    extract_emails, extract_urls, extract_phone_numbers,
    is_email_valid, is_url_valid, is_phone_valid, is_integer, is_float,
    celsius_to_fahrenheit, fahrenheit_to_celsius, meters_to_feet,
    feet_to_meters, kg_to_pounds, pounds_to_kg,
    get_file_extension, get_file_name, get_directory_name, join_paths,
    format_number, format_currency, format_percentage,
    set_union, set_intersection, set_difference,
    calculate_distance_2d, calculate_circle_area, calculate_rectangle_area,
    calculate_triangle_area
]

print(f"Created {len(tools)} tools total")

llm = init_chat_model(
        model_provider="openai",
        model=os.getenv("OPENAI_MODEL"),
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("OPENAI_BASE_URL"),
        temperature=0.7
    )
# llm = init_chat_model(
#         model_provider="ollama",
#         model=os.getenv("OLLAMA_MODEL"),
#         api_key=os.getenv("OLLAMAI_API_KEY"),
#         base_url=os.getenv("OLLAMA_BASE_URL"),
#         temperature=0.7,
#     )

llm_with_tools = llm.bind_tools(tools)

def chatbot(state: State):
    return {"messages": [llm_with_tools.invoke(state["messages"])]}

tool_node = ToolNode(tools=tools)


workflow = StateGraph(State)
workflow.add_node("chatbot", chatbot)
workflow.add_node("tools", tool_node)


workflow.add_edge(START, "chatbot")
workflow.add_conditional_edges(
    "chatbot",
    tools_condition,
)
workflow.add_edge("tools", "chatbot")

graph = workflow.compile()

if __name__ == "__main__":
    # test the graph
    import time
    start_time = time.time()
    user_input = "Calculate the least common multiple of 3 and 5"

    response = graph.invoke({"messages": [{"role": "user", "content": user_input}]})
    end_time = time.time()
    print(f"Time cost: {end_time - start_time} seconds")
    print("Assistant:", response["messages"][-1].content)
