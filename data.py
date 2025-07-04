import zipfile
import os

# Re-define paths after code state reset
zip_path = "9z52fv8ghd-1.zip"
extract_path = "dataset_extracted"

# Extract the zip file
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)

# List all files/folders inside the extracted directory
extracted_contents = []
for root, dirs, files in os.walk(extract_path):
    for name in files:
        extracted_contents.append(os.path.join(root, name))

extracted_contents[:20]  # Show first 20 items for brevity
