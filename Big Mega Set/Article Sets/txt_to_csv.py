import csv

# Define the file paths
input_file_path = 'amazon_cells_labelled.txt'
output_file_path = 'amazon_cells_labelled.csv'

# Open the input text file and output CSV file
with open(input_file_path, 'r', encoding='utf-8') as txt_file, open(output_file_path, 'w', newline='', encoding='utf-8') as csv_file:
    csv_writer = csv.writer(csv_file)
    
    # Write the header to the CSV file
    csv_writer.writerow(['Review', 'Label'])
    
    # Read each line from the text file
    for line in txt_file:
        # Split the line by the tab character
        review, label = line.strip().split('\t')
        # Write the review and label to the CSV file
        csv_writer.writerow([review, label])

print(f'File has been successfully converted and saved as {output_file_path}')
