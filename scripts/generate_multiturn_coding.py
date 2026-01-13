"""
Generate Multi-turn Code Editing Dataset

This script creates 100 conversations where users iteratively add requirements
to existing code, creating a natural multi-turn editing experience.
"""

import csv
import random
from typing import List, Dict

# Code templates and scenarios for multi-turn editing
CODE_SCENARIOS = [
    {
        "initial_request": "Create a simple Python function to calculate the area of a rectangle",
        "initial_code": """def calculate_area(length, width):
    return length * width""",
        "iterations": [
            {
                "user_request": "Add input validation to ensure length and width are positive numbers",
                "updated_code": """def calculate_area(length, width):
    if not isinstance(length, (int, float)) or not isinstance(width, (int, float)):
        raise TypeError("Length and width must be numbers")
    if length <= 0 or width <= 0:
        raise ValueError("Length and width must be positive")
    return length * width"""
            },
            {
                "user_request": "Add a docstring and type hints",
                "updated_code": """def calculate_area(length: float, width: float) -> float:
    \"\"\"
    Calculate the area of a rectangle.
    
    Args:
        length (float): The length of the rectangle
        width (float): The width of the rectangle
    
    Returns:
        float: The area of the rectangle
    
    Raises:
        TypeError: If length or width are not numbers
        ValueError: If length or width are not positive
    \"\"\"
    if not isinstance(length, (int, float)) or not isinstance(width, (int, float)):
        raise TypeError("Length and width must be numbers")
    if length <= 0 or width <= 0:
        raise ValueError("Length and width must be positive")
    return length * width"""
            },
            {
                "user_request": "Add unit tests for this function",
                "updated_code": """import unittest

def calculate_area(length: float, width: float) -> float:
    \"\"\"
    Calculate the area of a rectangle.
    
    Args:
        length (float): The length of the rectangle
        width (float): The width of the rectangle
    
    Returns:
        float: The area of the rectangle
    
    Raises:
        TypeError: If length or width are not numbers
        ValueError: If length or width are not positive
    \"\"\"
    if not isinstance(length, (int, float)) or not isinstance(width, (int, float)):
        raise TypeError("Length and width must be numbers")
    if length <= 0 or width <= 0:
        raise ValueError("Length and width must be positive")
    return length * width

class TestCalculateArea(unittest.TestCase):
    def test_valid_input(self):
        self.assertEqual(calculate_area(5, 3), 15)
        self.assertEqual(calculate_area(2.5, 4), 10)
    
    def test_invalid_type(self):
        with self.assertRaises(TypeError):
            calculate_area("5", 3)
    
    def test_negative_values(self):
        with self.assertRaises(ValueError):
            calculate_area(-5, 3)

if __name__ == "__main__":
    unittest.main()"""
            }
        ]
    },
    {
        "initial_request": "Write a Python class to represent a basic bank account",
        "initial_code": """class BankAccount:
    def __init__(self, balance=0):
        self.balance = balance
    
    def deposit(self, amount):
        self.balance += amount
    
    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
        else:
            print("Insufficient funds")
    
    def get_balance(self):
        return self.balance""",
        "iterations": [
            {
                "user_request": "Add account holder name and account number",
                "updated_code": """class BankAccount:
    def __init__(self, account_holder, account_number, balance=0):
        self.account_holder = account_holder
        self.account_number = account_number
        self.balance = balance
    
    def deposit(self, amount):
        self.balance += amount
    
    def withdraw(self, amount):
        if amount <= self.balance:
            self.balance -= amount
            return True
        else:
            print("Insufficient funds")
            return False
    
    def get_balance(self):
        return self.balance
    
    def get_account_info(self):
        return f"Account: {self.account_number}, Holder: {self.account_holder}, Balance: ${self.balance}" """
            },
            {
                "user_request": "Add transaction history tracking",
                "updated_code": """from datetime import datetime
from typing import List, Dict

class BankAccount:
    def __init__(self, account_holder, account_number, balance=0):
        self.account_holder = account_holder
        self.account_number = account_number
        self.balance = balance
        self.transaction_history: List[Dict] = []
    
    def _add_transaction(self, transaction_type, amount, description=""):
        transaction = {
            "timestamp": datetime.now(),
            "type": transaction_type,
            "amount": amount,
            "balance_after": self.balance,
            "description": description
        }
        self.transaction_history.append(transaction)
    
    def deposit(self, amount, description="Deposit"):
        if amount > 0:
            self.balance += amount
            self._add_transaction("DEPOSIT", amount, description)
            return True
        return False
    
    def withdraw(self, amount, description="Withdrawal"):
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            self._add_transaction("WITHDRAWAL", amount, description)
            return True
        else:
            print("Insufficient funds or invalid amount")
            return False
    
    def get_balance(self):
        return self.balance
    
    def get_account_info(self):
        return f"Account: {self.account_number}, Holder: {self.account_holder}, Balance: ${self.balance}"
    
    def get_transaction_history(self, limit=None):
        if limit:
            return self.transaction_history[-limit:]
        return self.transaction_history"""
            },
            {
                "user_request": "Add interest calculation and monthly statement generation",
                "updated_code": """from datetime import datetime, timedelta
from typing import List, Dict

class BankAccount:
    def __init__(self, account_holder, account_number, balance=0, interest_rate=0.01):
        self.account_holder = account_holder
        self.account_number = account_number
        self.balance = balance
        self.interest_rate = interest_rate
        self.transaction_history: List[Dict] = []
        self.last_interest_date = datetime.now()
    
    def _add_transaction(self, transaction_type, amount, description=""):
        transaction = {
            "timestamp": datetime.now(),
            "type": transaction_type,
            "amount": amount,
            "balance_after": self.balance,
            "description": description
        }
        self.transaction_history.append(transaction)
    
    def deposit(self, amount, description="Deposit"):
        if amount > 0:
            self.balance += amount
            self._add_transaction("DEPOSIT", amount, description)
            return True
        return False
    
    def withdraw(self, amount, description="Withdrawal"):
        if amount > 0 and amount <= self.balance:
            self.balance -= amount
            self._add_transaction("WITHDRAWAL", amount, description)
            return True
        else:
            print("Insufficient funds or invalid amount")
            return False
    
    def calculate_interest(self):
        days_since_last = (datetime.now() - self.last_interest_date).days
        if days_since_last >= 30:  # Monthly interest
            interest = self.balance * self.interest_rate
            self.balance += interest
            self._add_transaction("INTEREST", interest, "Monthly interest")
            self.last_interest_date = datetime.now()
            return interest
        return 0
    
    def generate_monthly_statement(self):
        end_date = datetime.now()
        start_date = end_date - timedelta(days=30)
        
        monthly_transactions = [
            t for t in self.transaction_history 
            if start_date <= t["timestamp"] <= end_date
        ]
        
        statement = {
            "account_number": self.account_number,
            "account_holder": self.account_holder,
            "statement_period": f"{start_date.date()} to {end_date.date()}",
            "opening_balance": monthly_transactions[0]["balance_after"] - monthly_transactions[0]["amount"] if monthly_transactions else self.balance,
            "closing_balance": self.balance,
            "transactions": monthly_transactions,
            "total_deposits": sum(t["amount"] for t in monthly_transactions if t["type"] == "DEPOSIT"),
            "total_withdrawals": sum(t["amount"] for t in monthly_transactions if t["type"] == "WITHDRAWAL"),
            "interest_earned": sum(t["amount"] for t in monthly_transactions if t["type"] == "INTEREST")
        }
        
        return statement
    
    def get_balance(self):
        return self.balance
    
    def get_account_info(self):
        return f"Account: {self.account_number}, Holder: {self.account_holder}, Balance: ${self.balance}"
    
    def get_transaction_history(self, limit=None):
        if limit:
            return self.transaction_history[-limit:]
        return self.transaction_history"""
            }
        ]
    }
]

# Additional scenario templates
SCENARIO_TEMPLATES = [
    {
        "domain": "web_scraping",
        "initial": "Create a simple web scraper for product prices",
        "evolutions": ["Add error handling", "Add rate limiting", "Add data export to CSV", "Add proxy support"]
    },
    {
        "domain": "data_analysis",
        "initial": "Write a function to analyze sales data",
        "evolutions": ["Add data visualization", "Add statistical analysis", "Add trend prediction", "Add report generation"]
    },
    {
        "domain": "api_development",
        "initial": "Create a REST API endpoint for user management",
        "evolutions": ["Add authentication", "Add input validation", "Add database integration", "Add rate limiting"]
    },
    {
        "domain": "file_processing",
        "initial": "Write a script to process CSV files",
        "evolutions": ["Add error handling", "Add progress tracking", "Add multiple format support", "Add parallel processing"]
    },
    {
        "domain": "machine_learning",
        "initial": "Create a simple linear regression model",
        "evolutions": ["Add data preprocessing", "Add model evaluation", "Add cross-validation", "Add hyperparameter tuning"]
    }
]

def generate_code_progression(domain: str, language: str = "Python") -> Dict:
    """Generate a code progression for a given domain."""
    
    if domain == "calculator":
        return {
            "initial_request": f"Create a simple {language} calculator class",
            "initial_code": """class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        return a / b""",
            "iterations": [
                {
                    "user_request": "Add error handling for division by zero",
                    "updated_code": """class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b"""
                },
                {
                    "user_request": "Add advanced operations like power and square root",
                    "updated_code": """import math

class Calculator:
    def add(self, a, b):
        return a + b
    
    def subtract(self, a, b):
        return a - b
    
    def multiply(self, a, b):
        return a * b
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    
    def power(self, base, exponent):
        return base ** exponent
    
    def square_root(self, number):
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        return math.sqrt(number)
    
    def factorial(self, n):
        if n < 0:
            raise ValueError("Cannot calculate factorial of negative number")
        if n == 0 or n == 1:
            return 1
        return n * self.factorial(n - 1)"""
                },
                {
                    "user_request": "Add calculation history and memory functions",
                    "updated_code": """import math
from datetime import datetime
from typing import List, Dict

class Calculator:
    def __init__(self):
        self.history: List[Dict] = []
        self.memory = 0
    
    def _log_operation(self, operation, operands, result):
        self.history.append({
            "timestamp": datetime.now(),
            "operation": operation,
            "operands": operands,
            "result": result
        })
    
    def add(self, a, b):
        result = a + b
        self._log_operation("add", [a, b], result)
        return result
    
    def subtract(self, a, b):
        result = a - b
        self._log_operation("subtract", [a, b], result)
        return result
    
    def multiply(self, a, b):
        result = a * b
        self._log_operation("multiply", [a, b], result)
        return result
    
    def divide(self, a, b):
        if b == 0:
            raise ValueError("Cannot divide by zero")
        result = a / b
        self._log_operation("divide", [a, b], result)
        return result
    
    def power(self, base, exponent):
        result = base ** exponent
        self._log_operation("power", [base, exponent], result)
        return result
    
    def square_root(self, number):
        if number < 0:
            raise ValueError("Cannot calculate square root of negative number")
        result = math.sqrt(number)
        self._log_operation("square_root", [number], result)
        return result
    
    def factorial(self, n):
        if n < 0:
            raise ValueError("Cannot calculate factorial of negative number")
        if n == 0 or n == 1:
            result = 1
        else:
            result = n * self.factorial(n - 1)
        self._log_operation("factorial", [n], result)
        return result
    
    def memory_store(self, value):
        self.memory = value
        return f"Stored {value} in memory"
    
    def memory_recall(self):
        return self.memory
    
    def memory_clear(self):
        self.memory = 0
        return "Memory cleared"
    
    def get_history(self, limit=None):
        if limit:
            return self.history[-limit:]
        return self.history
    
    def clear_history(self):
        self.history.clear()
        return "History cleared" """
                }
            ]
        }
    
    elif domain == "todo_app":
        return {
            "initial_request": f"Create a simple {language} to-do list class",
            "initial_code": """class TodoList:
    def __init__(self):
        self.tasks = []
    
    def add_task(self, task):
        self.tasks.append(task)
    
    def remove_task(self, task):
        if task in self.tasks:
            self.tasks.remove(task)
    
    def list_tasks(self):
        return self.tasks""",
            "iterations": [
                {
                    "user_request": "Add task completion status and due dates",
                    "updated_code": """from datetime import datetime
from typing import List, Dict, Optional

class TodoList:
    def __init__(self):
        self.tasks: List[Dict] = []
        self.next_id = 1
    
    def add_task(self, title: str, due_date: Optional[str] = None):
        task = {
            "id": self.next_id,
            "title": title,
            "completed": False,
            "created_at": datetime.now(),
            "due_date": datetime.strptime(due_date, "%Y-%m-%d") if due_date else None,
            "completed_at": None
        }
        self.tasks.append(task)
        self.next_id += 1
        return task["id"]
    
    def complete_task(self, task_id: int):
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = True
                task["completed_at"] = datetime.now()
                return True
        return False
    
    def remove_task(self, task_id: int):
        self.tasks = [task for task in self.tasks if task["id"] != task_id]
    
    def list_tasks(self, show_completed=True):
        if show_completed:
            return self.tasks
        return [task for task in self.tasks if not task["completed"]]
    
    def get_overdue_tasks(self):
        now = datetime.now()
        return [task for task in self.tasks 
                if task["due_date"] and task["due_date"] < now and not task["completed"]]"""
                },
                {
                    "user_request": "Add priority levels and categories",
                    "updated_code": """from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class TodoList:
    def __init__(self):
        self.tasks: List[Dict] = []
        self.next_id = 1
        self.categories = set()
    
    def add_task(self, title: str, due_date: Optional[str] = None, 
                 priority: Priority = Priority.MEDIUM, category: str = "General"):
        task = {
            "id": self.next_id,
            "title": title,
            "completed": False,
            "created_at": datetime.now(),
            "due_date": datetime.strptime(due_date, "%Y-%m-%d") if due_date else None,
            "completed_at": None,
            "priority": priority,
            "category": category
        }
        self.tasks.append(task)
        self.categories.add(category)
        self.next_id += 1
        return task["id"]
    
    def complete_task(self, task_id: int):
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = True
                task["completed_at"] = datetime.now()
                return True
        return False
    
    def remove_task(self, task_id: int):
        self.tasks = [task for task in self.tasks if task["id"] != task_id]
    
    def list_tasks(self, show_completed=True, category=None, priority=None):
        filtered_tasks = self.tasks
        
        if not show_completed:
            filtered_tasks = [task for task in filtered_tasks if not task["completed"]]
        
        if category:
            filtered_tasks = [task for task in filtered_tasks if task["category"] == category]
        
        if priority:
            filtered_tasks = [task for task in filtered_tasks if task["priority"] == priority]
        
        # Sort by priority (urgent first) then by due date
        return sorted(filtered_tasks, 
                     key=lambda x: (x["priority"].value, x["due_date"] or datetime.max), 
                     reverse=True)
    
    def get_overdue_tasks(self):
        now = datetime.now()
        return [task for task in self.tasks 
                if task["due_date"] and task["due_date"] < now and not task["completed"]]
    
    def get_categories(self):
        return sorted(self.categories)
    
    def get_tasks_by_priority(self, priority: Priority):
        return [task for task in self.tasks if task["priority"] == priority]"""
                },
                {
                    "user_request": "Add file persistence and search functionality",
                    "updated_code": """from datetime import datetime
from typing import List, Dict, Optional
from enum import Enum
import json
import os

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

class TodoList:
    def __init__(self, filename: str = "todos.json"):
        self.tasks: List[Dict] = []
        self.next_id = 1
        self.categories = set()
        self.filename = filename
        self.load_from_file()
    
    def add_task(self, title: str, due_date: Optional[str] = None, 
                 priority: Priority = Priority.MEDIUM, category: str = "General"):
        task = {
            "id": self.next_id,
            "title": title,
            "completed": False,
            "created_at": datetime.now().isoformat(),
            "due_date": due_date,
            "completed_at": None,
            "priority": priority.value,
            "category": category
        }
        self.tasks.append(task)
        self.categories.add(category)
        self.next_id += 1
        self.save_to_file()
        return task["id"]
    
    def complete_task(self, task_id: int):
        for task in self.tasks:
            if task["id"] == task_id:
                task["completed"] = True
                task["completed_at"] = datetime.now().isoformat()
                self.save_to_file()
                return True
        return False
    
    def remove_task(self, task_id: int):
        self.tasks = [task for task in self.tasks if task["id"] != task_id]
        self.save_to_file()
    
    def search_tasks(self, query: str):
        query = query.lower()
        return [task for task in self.tasks 
                if query in task["title"].lower() or query in task["category"].lower()]
    
    def list_tasks(self, show_completed=True, category=None, priority=None):
        filtered_tasks = self.tasks
        
        if not show_completed:
            filtered_tasks = [task for task in filtered_tasks if not task["completed"]]
        
        if category:
            filtered_tasks = [task for task in filtered_tasks if task["category"] == category]
        
        if priority:
            filtered_tasks = [task for task in filtered_tasks if task["priority"] == priority.value]
        
        return sorted(filtered_tasks, 
                     key=lambda x: (x["priority"], x["due_date"] or "9999-12-31"), 
                     reverse=True)
    
    def save_to_file(self):
        data = {
            "tasks": self.tasks,
            "next_id": self.next_id,
            "categories": list(self.categories)
        }
        with open(self.filename, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_from_file(self):
        if os.path.exists(self.filename):
            try:
                with open(self.filename, 'r') as f:
                    data = json.load(f)
                    self.tasks = data.get("tasks", [])
                    self.next_id = data.get("next_id", 1)
                    self.categories = set(data.get("categories", []))
            except (json.JSONDecodeError, FileNotFoundError):
                pass
    
    def get_statistics(self):
        total_tasks = len(self.tasks)
        completed_tasks = len([task for task in self.tasks if task["completed"]])
        overdue_tasks = len(self.get_overdue_tasks())
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": total_tasks - completed_tasks,
            "overdue_tasks": overdue_tasks,
            "completion_rate": (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        }
    
    def get_overdue_tasks(self):
        now = datetime.now().date()
        overdue = []
        for task in self.tasks:
            if (task["due_date"] and not task["completed"] and 
                datetime.fromisoformat(task["due_date"]).date() < now):
                overdue.append(task)
        return overdue"""
                }
            ]
        }
    
    # Add more domain generators as needed
    return {"initial_request": f"Create a {domain} application", "initial_code": "# Code here", "iterations": []}

def generate_conversation(conv_id: str, scenario: Dict) -> List[Dict]:
    """Generate a multi-turn conversation from a scenario."""
    conversation = []
    turn = 1
    
    # Initial request
    conversation.append({
        "conversation_id": conv_id,
        "turn_index": turn,
        "role": "user",
        "message": scenario["initial_request"]
    })
    turn += 1
    
    # Initial code response
    conversation.append({
        "conversation_id": conv_id,
        "turn_index": turn,
        "role": "assistant", 
        "message": f"Here's a basic implementation:\n\n```python\n{scenario['initial_code']}\n```\n\nThis provides the core functionality you requested. Would you like me to add any additional features?"
    })
    turn += 1
    
    # Iterative improvements
    for iteration in scenario["iterations"]:
        # User requests enhancement
        conversation.append({
            "conversation_id": conv_id,
            "turn_index": turn,
            "role": "user",
            "message": iteration["user_request"]
        })
        turn += 1
        
        # Assistant provides updated code
        conversation.append({
            "conversation_id": conv_id,
            "turn_index": turn,
            "role": "assistant",
            "message": f"I've updated the code to include {iteration['user_request'].lower()}:\n\n```python\n{iteration['updated_code']}\n```\n\nThe code now has enhanced functionality. Is there anything else you'd like me to add or modify?"
        })
        turn += 1
    
    return conversation

def generate_all_conversations() -> List[Dict]:
    """Generate all 100 conversations."""
    conversations = []
    
    # Use predefined scenarios
    for i, scenario in enumerate(CODE_SCENARIOS):
        conv_id = f"CD{i+1:03d}"
        conversations.extend(generate_conversation(conv_id, scenario))
    
    # Generate additional scenarios
    domains = ["calculator", "todo_app", "password_generator", "file_organizer", "weather_app", 
              "inventory_system", "chat_bot", "expense_tracker", "note_taking", "timer_app"]
    
    scenario_count = len(CODE_SCENARIOS)
    
    for i in range(scenario_count, 100):
        conv_id = f"CD{i+1:03d}"
        domain = domains[i % len(domains)]
        scenario = generate_code_progression(domain)
        conversations.extend(generate_conversation(conv_id, scenario))
    
    return conversations

def save_to_csv(conversations: List[Dict], filename: str):
    """Save conversations to CSV file."""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['conversation_id', 'turn_index', 'role', 'message']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for conv in conversations:
            writer.writerow(conv)

def main():
    print("Generating 100 multi-turn code editing conversations...")
    conversations = generate_all_conversations()
    
    output_file = "dataset/benchmark_synthetic_dataset/coding/coding_new.csv"
    save_to_csv(conversations, output_file)
    
    print(f"Generated {len(conversations)} conversation turns")
    print(f"Saved to {output_file}")
    
    # Count unique conversations
    unique_convs = len(set(conv["conversation_id"] for conv in conversations))
    print(f"Total unique conversations: {unique_convs}")

if __name__ == "__main__":
    main()
