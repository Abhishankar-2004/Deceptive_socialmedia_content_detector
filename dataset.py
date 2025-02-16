import os
import pandas as pd

# Load the Excel file
excel_file = r'output_file.csv'  # Use raw string (r) to avoid escape character issues

# Read the Excel file
df = pd.read_excel(excel_file)

# Save as CSV
output_file = 'output_file.csv'  # Specify the output file path
df.to_csv(output_file, index=False)  # Write DataFrame to CSV

# Check if the CSV file exists
if os.path.exists(output_file):
    print(f"File '{output_file}' loaded successfully!")
else:
    print(f"File '{output_file}' not found. Please check the file path.")


