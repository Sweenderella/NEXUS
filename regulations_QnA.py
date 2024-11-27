import os
import json
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
import torch

# Initialize models
model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")  # For similarity scoring

def generate_query_embedding(query):
    """
    Generate an embedding for the query using SentenceTransformer.
    """
    return embedding_model.encode(query)

def split_context_into_chunks(context, max_chunk_size):
    """
    Split context into smaller chunks for tokenization.
    """
    context_tokens = tokenizer.encode(context, truncation=False)
    chunks = [
        tokenizer.decode(context_tokens[i:i + max_chunk_size], skip_special_tokens=True)
        for i in range(0, len(context_tokens), max_chunk_size)
    ]
    return chunks

def find_relevant_section(query, json_data):
    """
    Find the most relevant section for a query based on cosine similarity.

    Args:
        query (str): The query to find the relevant section for.
        json_data (dict): The JSON data containing chapters, sections, and embeddings.

    Returns:
        dict: A dictionary with the best context and its associated metadata.
    """
    max_similarity = 0
    best_context = ""
    best_metadata = {}

    # Generate query embedding
    query_embedding = generate_query_embedding(query)
    print("Query embedding:", query_embedding[:5])  # Debug: Show first 5 values

    for chapter in json_data.get('chapters', []):
        for section in chapter.get('sections', []):
            # Safely access content_embedding and section_title_embedding
            content_embedding = section.get('content_embedding')
            section_embedding = section.get('section_title_embedding')
            
            # Skip sections that are missing embeddings
            if content_embedding and 'vector' in content_embedding:
                embedding_vector = content_embedding['vector']
                similarity = cosine_similarity([query_embedding], [embedding_vector])[0][0]
                print(f"Similarity with section '{section.get('title')}': {similarity}")  # Debug

                if similarity > max_similarity:
                    max_similarity = similarity
                    best_context = section.get('content', '') or section.get('title', '')
                    best_metadata = {
                        "chapter_title": chapter.get("chapter_title", "Unknown Chapter"),
                        "section_title": section.get("title", "Unknown Section"),
                        "section_number": section.get("section_number", "N/A"),
                        "page_number": section.get("page_number", "N/A"),
                    }

    print(f"Best context found: {best_context[:100]}")  # Debug: Show first 100 chars of context
    print(f"Metadata of best context: {best_metadata}")  # Debug: Show metadata of the best context

    return {
        "best_context": best_context,
        "metadata": best_metadata
    }


def generate_answer_with_flan_t5(query, context, max_chunk_size=500):
    """
    Generate a detailed answer to a query using FLAN-T5 with context chunks.
    """
    context_chunks = split_context_into_chunks(context, max_chunk_size)
    query_embedding = embedding_model.encode(query)
    
    # Select the best chunk
    best_chunk, max_similarity = "", 0
    for chunk in context_chunks:
        chunk_embedding = embedding_model.encode(chunk)
        similarity = util.cos_sim(query_embedding, chunk_embedding).item()
        if similarity > max_similarity:
            best_chunk, max_similarity = chunk, similarity

    # Generate answer
    prompt = (
        f"Question: {query}\n"
        f"Context: {best_chunk}\n"
        f"Answer:"
    )
    inputs = tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True).to(model.device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=200,
        num_beams=5,
        early_stopping=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

def load_json_files(folder_path):
    """
    Load all JSON files from a specified folder.
    """
    data = []
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.json'):
            with open(os.path.join(folder_path, file_name), 'r') as file:
                data.append(json.load(file))
    return data



def symbolic_reasoning_regulations(compliance_query):    # Folder containing JSON files
    folder_path = "G:/USA Projects/Co2 data/dataset/Generic QnA/carbon regulations/Regulations_Chapterwise_Embedding_files"

    # Load JSON data
    all_data = load_json_files(folder_path)

    # Example query
    query = compliance_query
    print('QUery posted by the Neural Predictor is ----> ')
    print(query)
#    "What is the modeling of consumer purchase decision?"

    # Iterate over JSON files to find the answer
    for json_file in all_data:
        result = find_relevant_section(query, json_file)
        proper_context = result["best_context"]
        metadata = result["metadata"]

        if proper_context:
            answer = generate_answer_with_flan_t5(query, proper_context)
            print('------------Based on your query----------')
            print(compliance_query)
            print('System has referred the regulations end to end and identified relevant context as follows ')
            print(f"Context:\n{proper_context}")  # Show first 500 characters of context
            print ('System has identified relevant context for your query and generated appropriate answer ---> ')
           
            print(f"Answer:->>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n{answer}")
            print('The details of the answer and relevant context can be found you in Federal Environment regulations as per chapter, section details as follows:')
            
            print(f"Metadata:\n{metadata}")
        else:
            print(f"No context found for query: {query}")

