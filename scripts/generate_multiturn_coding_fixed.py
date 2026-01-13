"""
Generate Multi-turn Code Editing Dataset

This script creates 100 conversations where users iteratively add requirements
to existing code, creating a natural multi-turn editing experience.
"""

import csv
import random
from typing import List, Dict

def generate_simple_scenarios():
    """Generate simple coding scenarios that can be expanded iteratively."""
    
    scenarios = []
    
    # Scenario 1: Rectangle Area Calculator
    scenarios.append({
        "initial_request": "Create a simple Python function to calculate the area of a rectangle",
        "initial_code": "def calculate_area(length, width):\n    return length * width",
        "iterations": [
            {
                "user_request": "Add input validation to ensure length and width are positive numbers",
                "updated_code": "def calculate_area(length, width):\n    if not isinstance(length, (int, float)) or not isinstance(width, (int, float)):\n        raise TypeError(\"Length and width must be numbers\")\n    if length <= 0 or width <= 0:\n        raise ValueError(\"Length and width must be positive\")\n    return length * width"
            },
            {
                "user_request": "Add type hints and a docstring",
                "updated_code": "def calculate_area(length: float, width: float) -> float:\n    \"\"\"\n    Calculate the area of a rectangle.\n    \n    Args:\n        length (float): The length of the rectangle\n        width (float): The width of the rectangle\n    \n    Returns:\n        float: The area of the rectangle\n    \n    Raises:\n        TypeError: If length or width are not numbers\n        ValueError: If length or width are not positive\n    \"\"\"\n    if not isinstance(length, (int, float)) or not isinstance(width, (int, float)):\n        raise TypeError(\"Length and width must be numbers\")\n    if length <= 0 or width <= 0:\n        raise ValueError(\"Length and width must be positive\")\n    return length * width"
            }
        ]
    })
    
    # Scenario 2: Simple Calculator
    scenarios.append({
        "initial_request": "Create a basic calculator class with add, subtract, multiply, and divide methods",
        "initial_code": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n    \n    def multiply(self, a, b):\n        return a * b\n    \n    def divide(self, a, b):\n        return a / b",
        "iterations": [
            {
                "user_request": "Add error handling for division by zero",
                "updated_code": "class Calculator:\n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n    \n    def multiply(self, a, b):\n        return a * b\n    \n    def divide(self, a, b):\n        if b == 0:\n            raise ValueError(\"Cannot divide by zero\")\n        return a / b"
            },
            {
                "user_request": "Add memory functionality to store and recall values",
                "updated_code": "class Calculator:\n    def __init__(self):\n        self.memory = 0\n    \n    def add(self, a, b):\n        return a + b\n    \n    def subtract(self, a, b):\n        return a - b\n    \n    def multiply(self, a, b):\n        return a * b\n    \n    def divide(self, a, b):\n        if b == 0:\n            raise ValueError(\"Cannot divide by zero\")\n        return a / b\n    \n    def memory_store(self, value):\n        self.memory = value\n    \n    def memory_recall(self):\n        return self.memory\n    \n    def memory_clear(self):\n        self.memory = 0"
            }
        ]
    })
    
    return scenarios

def generate_additional_scenario(domain: str, index: int) -> Dict:
    """Generate additional scenarios based on domain."""
    
    scenarios_by_domain = {
        "todo": {
            "initial_request": "Create a simple to-do list class",
            "initial_code": "class TodoList:\n    def __init__(self):\n        self.tasks = []\n    \n    def add_task(self, task):\n        self.tasks.append(task)\n    \n    def remove_task(self, task):\n        if task in self.tasks:\n            self.tasks.remove(task)\n    \n    def list_tasks(self):\n        return self.tasks",
            "iterations": [
                {
                    "user_request": "Add task completion status",
                    "updated_code": "class TodoList:\n    def __init__(self):\n        self.tasks = []\n    \n    def add_task(self, task):\n        self.tasks.append({'task': task, 'completed': False})\n    \n    def complete_task(self, task_name):\n        for task in self.tasks:\n            if task['task'] == task_name:\n                task['completed'] = True\n                break\n    \n    def list_tasks(self, show_completed=True):\n        if show_completed:\n            return self.tasks\n        return [task for task in self.tasks if not task['completed']]"
                },
                {
                    "user_request": "Add priority levels for tasks",
                    "updated_code": "class TodoList:\n    def __init__(self):\n        self.tasks = []\n    \n    def add_task(self, task, priority='medium'):\n        self.tasks.append({\n            'task': task, \n            'completed': False, \n            'priority': priority\n        })\n    \n    def complete_task(self, task_name):\n        for task in self.tasks:\n            if task['task'] == task_name:\n                task['completed'] = True\n                break\n    \n    def list_tasks(self, show_completed=True, priority=None):\n        filtered_tasks = self.tasks\n        \n        if not show_completed:\n            filtered_tasks = [task for task in filtered_tasks if not task['completed']]\n        \n        if priority:\n            filtered_tasks = [task for task in filtered_tasks if task['priority'] == priority]\n        \n        return sorted(filtered_tasks, key=lambda x: {'high': 3, 'medium': 2, 'low': 1}[x['priority']], reverse=True)"
                }
            ]
        },
        "weather": {
            "initial_request": "Create a simple weather data class",
            "initial_code": "class WeatherData:\n    def __init__(self, temperature, humidity):\n        self.temperature = temperature\n        self.humidity = humidity\n    \n    def get_temperature(self):\n        return self.temperature\n    \n    def get_humidity(self):\n        return self.humidity",
            "iterations": [
                {
                    "user_request": "Add weather conditions and wind speed",
                    "updated_code": "class WeatherData:\n    def __init__(self, temperature, humidity, condition='clear', wind_speed=0):\n        self.temperature = temperature\n        self.humidity = humidity\n        self.condition = condition\n        self.wind_speed = wind_speed\n    \n    def get_temperature(self):\n        return self.temperature\n    \n    def get_humidity(self):\n        return self.humidity\n    \n    def get_condition(self):\n        return self.condition\n    \n    def get_wind_speed(self):\n        return self.wind_speed\n    \n    def get_summary(self):\n        return f'Temperature: {self.temperature}°C, Humidity: {self.humidity}%, Condition: {self.condition}, Wind: {self.wind_speed} km/h'"
                },
                {
                    "user_request": "Add temperature conversion methods",
                    "updated_code": "class WeatherData:\n    def __init__(self, temperature, humidity, condition='clear', wind_speed=0, temp_unit='C'):\n        self.temperature = temperature\n        self.humidity = humidity\n        self.condition = condition\n        self.wind_speed = wind_speed\n        self.temp_unit = temp_unit\n    \n    def get_temperature(self, unit=None):\n        if unit is None:\n            unit = self.temp_unit\n        \n        if self.temp_unit == 'C' and unit == 'F':\n            return (self.temperature * 9/5) + 32\n        elif self.temp_unit == 'F' and unit == 'C':\n            return (self.temperature - 32) * 5/9\n        else:\n            return self.temperature\n    \n    def get_humidity(self):\n        return self.humidity\n    \n    def get_condition(self):\n        return self.condition\n    \n    def get_wind_speed(self):\n        return self.wind_speed\n    \n    def get_summary(self, temp_unit=None):\n        temp = self.get_temperature(temp_unit or self.temp_unit)\n        unit_symbol = temp_unit or self.temp_unit\n        return f'Temperature: {temp:.1f}°{unit_symbol}, Humidity: {self.humidity}%, Condition: {self.condition}, Wind: {self.wind_speed} km/h'"
                }
            ]
        }
    }
    
    return scenarios_by_domain.get(domain, scenarios_by_domain["todo"])

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
    predefined_scenarios = generate_simple_scenarios()
    
    for i, scenario in enumerate(predefined_scenarios):
        conv_id = f"CD{i+1:03d}"
        conversations.extend(generate_conversation(conv_id, scenario))
    
    # Generate additional scenarios
    domains = ["todo", "weather", "calculator", "password", "file", "inventory", "chat", "expense", "note", "timer"]
    
    for i in range(len(predefined_scenarios), 100):
        conv_id = f"CD{i+1:03d}"
        domain = domains[i % len(domains)]
        scenario = generate_additional_scenario(domain, i)
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
