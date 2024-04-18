import json
import csv

# Importing custom functions for processing conversation data
from source_cleanup import source_conversation_data, context_conversation_data

# Define the path where the data files are stored
data_path = '../../data'

# Step 0: Declarations 
equal_data = []
non_equal_data = []
equal_data_conversations_object = []
non_equal_data_conversations_object = []

# Step 1: Open and read the non-equal data CSV file
with open(f'{data_path}/cleaned_non_equal.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    for line in reader:
        non_equal_data.append(line)  # Append each row to the list

# Step 2: Open and read the equal data CSV file
with open(f'{data_path}/cleaned_equal.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',', quotechar='"')
    for line in reader:
        equal_data.append(line)  # Append each row to the list


# Step 3: Process each row from the equal data except the header row
for data in equal_data[1:]:
    data_object = {}  # Create a dictionary for each conversation
    data_object['id'] = data[0]  # Store the unique identifier
    data_object['prev_context_conversation'] = context_conversation_data(data[1])  # Process and store the previous context
    data_object['ai_response'] = data[2]  # Store the AI response
    data_object['human_response'] = data[3]  # Store the human response
    data_object['source_conversation'] = source_conversation_data(data[4])  # Process and store the source conversation data
    equal_data_conversations_object.append(data_object)  # Append the dictionary to the list

# Step 4: Write the structured data into a JSON file for equal data
with open(f'{data_path}/structured_equal.json', 'w') as f:
    json.dump(equal_data_conversations_object, f, indent=4)  # Serialize the list of dictionaries to JSON


# Step 5: Process each row from the non-equal data except the header row
for data in non_equal_data[1:]:
    data_object = {}
    data_object['id'] = data[0]
    data_object['prev_context_conversation'] = context_conversation_data(data[1])
    data_object['ai_response'] = data[2]
    data_object['human_response'] = data[3]
    data_object['source_conversation'] = source_conversation_data(data[4])
    non_equal_data_conversations_object.append(data_object)

# Step 6: Write the structured data into a JSON file for non-equal data
with open(f'{data_path}/structured_non_equal.json', 'w') as f:
    json.dump(non_equal_data_conversations_object, f, indent=4)
