import json
from collections import defaultdict

# Step 1: Read the JSON file
with open('CoT_prompts.json', 'r') as file:
    data = json.load(file)

# Step 2: Group data by 'key'
grouped_data = defaultdict(list)
for item in data:
    grouped_data[item['key']].append(item)

# Step 3: Split each group into training and testing
train_data = []
test_data = []

for key, items in grouped_data.items():
    mid_point = len(items) // 2
    test_data.extend(items[:mid_point])
    train_data.extend(items[mid_point:])

# Step 4: Write the split data back to JSON files
with open('train_CoT_prompts.json', 'w') as file:
    json.dump(train_data, file, indent=4)

with open('test_CoT_prompts.json', 'w') as file:
    json.dump(test_data, file, indent=4)
