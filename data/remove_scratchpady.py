import json
import re

# Path to the JSON file
file_path = '/colab/sleeperagents/sleeper-agents-paper/test_CoT_prompts.json'

# Function to remove content between <scratchpad> tags
def remove_scratchpad_content(data):
    pattern = r'<scratchpad>.*?</scratchpad>'
    return re.sub(pattern, '', data, flags=re.DOTALL)

# Read the JSON file
with open(file_path, 'r') as file:
    data = json.load(file)

# Iterate through the list and process each "output" field
for item in data:
    if 'output' in item:
        item['output'] = remove_scratchpad_content(item['output'])

# You can either print the modified data or write it back to a file
print(json.dumps(data, indent=4))

save_file_path = 'test_distilled_CoT_prompts.json'
# Optionally, write the modified data back to the file or another file
with open(save_file_path, 'w') as file:
    json.dump(data, file, indent=4)
