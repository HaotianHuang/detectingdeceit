import json
import re
import sys
import os

# Check if the input file path is provided as an argument
if len(sys.argv) < 2:
    print("Usage: python script.py <input_file_path>")
    sys.exit(1)

# Get the input file path from command-line arguments
input_file_path = sys.argv[1]
# Get the directory of the input file
input_dir = os.path.dirname(input_file_path)
# Generate the output file path with .py extension in the same directory
output_file_name = re.sub(r'\.jsonl$', '.py', os.path.basename(input_file_path))
output_file_path = os.path.join(input_dir, output_file_name)

# Open the input file and output file
with open(input_file_path, 'r') as input_file, open(output_file_path, 'w') as output_file:
    line_number = 0
    for line in input_file:
        line_number += 1
        # Try to load JSON data from the line
        try:
            data = json.loads(line)
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON on line {line_number}: {e}")
            continue  # Skip to the next line if there's a decoding error

        # Check if 'predict' key exists in the JSON data
        if 'predict' in data:
            # Append "</code>" to the end of the 'predict' value
            data['predict'] += "</code>"
            # Find all <code></code> blocks
            code_blocks = re.findall(r'<code>(.*?)</code>', data['predict'], re.DOTALL)

            # Identify the longest code block
            longest_block = max(code_blocks, key=len, default="")

            # Strip "tags:" and "<code>" from the code block
            longest_block = re.sub(r'^\s*tags:\s*', '', longest_block, flags=re.IGNORECASE)
            longest_block = re.sub(r'^<code>', '', longest_block)

            # Write the longest code block to the output file
            if longest_block:
                output_file.write(f"# Longest code block from line {line_number}:\n")
                output_file.write(longest_block.strip() + "\n\n")
