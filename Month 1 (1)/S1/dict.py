# Creating and manipulating a dictionary
student_scores = {
    "Alice": 85,
    "Bob": 92,
    "Charlie": 78
}

# Adding a new entry
student_scores["Diana"] = 95

# Dictionary Comprehension: Create a new dict with only passing grades (>80)
passing_students = {name: score for name, score in student_scores.items() if score > 80}

print(f"All scores: {student_scores}")
print(f"Passing students: {passing_students}")