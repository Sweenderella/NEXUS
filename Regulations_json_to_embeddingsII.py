import json
import os
from sentence_transformers import SentenceTransformer

# Directory where the JSON files are stored
input_folder = "G:/USA Projects/Co2 data/dataset/Generic QnA/carbon regulations/Chapter 5"  # Replace with your folder path

# Directory to save the output JSON files
output_folder = "G:/USA Projects/Co2 data/dataset/Generic QnA/carbon regulations/Chapter 5"  # Replace with your desired output folder path

import os
import json
from sentence_transformers import SentenceTransformer

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Function to generate embeddings and metadata
def generate_embedding(text, metadata_type, **kwargs):
    if not text:  # Handle missing or empty text
        return None
    return {
        "vector": model.encode(text).tolist(),
        "metadata": {
            "type": metadata_type,
            **kwargs
        }
    }

# Recursive function to process hierarchy and attach metadata
def process_hierarchy(entity, level_type, parent_metadata=None):
    """
    Processes hierarchical entities like chapters, sections, and subsections.
    Generates embeddings and saves them directly at the respective level.

    Args:
    - entity (dict): The dictionary object at the current hierarchy level.
    - level_type (str): The type of the entity ('chapter', 'section', 'subsection', etc.).
    - parent_metadata (dict): Metadata inherited from the parent level.
    """
    # Prepare metadata for the current level
    current_metadata = {
        **(parent_metadata or {}),
        f"{level_type}_title": entity.get("title"),
        f"{level_type}_number": entity.get(f"{level_type}_number"),
        "page_number": entity.get("page_number"),
    }

    # Generate and save embeddings for 'content'
    if 'content' in entity:
        entity['content_embedding'] = generate_embedding(
            entity['content'],
            "content",
            **current_metadata
        )

    # Generate and save embeddings for 'title'
    if 'title' in entity:
        entity['title_embedding'] = generate_embedding(
            entity['title'],
            f"{level_type}_title",
            **current_metadata
        )

    # Process subsections or nested entities
    if level_type == "chapter":
        for section in entity.get('sections', []):
            process_hierarchy(section, "section", current_metadata)
    elif level_type == "section":
        for subsection in entity.get('subsections', []):
            process_hierarchy(subsection, "subsection", current_metadata)

# Input and output folder paths
#input_folder = "path/to/input_folder"
#output_folder = "path/to/output_folder"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Process each JSON file in the input folder
for filename in os.listdir(input_folder):
    if filename.endswith(".json"):
        input_json_path = os.path.join(input_folder, filename)

        try:
            # Load the JSON data
            with open(input_json_path, 'r') as file:
                data = json.load(file)

            # Process each chapter individually
            for chapter in data.get('chapters', []):
                process_hierarchy(chapter, "chapter")

            # Save the enriched data with embeddings to a new JSON file
            output_json_path = os.path.join(output_folder, f"Regulations_embeddings_{filename}")
            with open(output_json_path, 'w') as outfile:
                json.dump(data, outfile, indent=2)

            print(f"Processed file saved: {output_json_path}")

        except Exception as e:
            print(f"Error processing file {filename}: {e}")
