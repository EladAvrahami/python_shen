# Define a dictionary
person = {
    "name": "Alice",
    "age": 25,
    "city": "New York"
}

# Access a value using a key
print("Name:", person["name"])  # Output: Name: Alice

# Add a new key-value pair
person["job"] = "Engineer"
print("Updated dictionary:", person)

# Modify an existing value
person["age"] = 26
print("Modified dictionary:", person)

# Remove a key-value pair
del person["city"]
print("Dictionary after deletion:", person)

# Iterate through the dictionary
for key, value in person.items():
    print(f"{key}: {value}")