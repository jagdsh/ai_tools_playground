import time
import torch
from PIL import Image
from paddleocr import PaddleOCR
from colpali_engine.models import ColPali, ColPaliProcessor

# --- Configuration ---
IMAGE_PATH = "run_pod_machines.png" # Replace with a complex document image (e.g., a PDF page with tables)
TEST_QUERY = "What is the total revenue reported in the Q3 table?"

def demo_paddleocr():
    print("--- Starting PaddleOCR ---")

    ocr = PaddleOCR(use_angle_cls=True, lang='en', show_log=False)
    
    start_time = time.time()
    
    result = ocr.ocr(IMAGE_PATH, cls=True)
    
    extraction_time = time.time() - start_time

    extracted_text = [line[1][0] for res in result for line in res]

    print(f"PaddleOCR Extraction Time: {extraction_time:.4f} seconds")
    print(f"Extracted {len(extracted_text)} lines of text.\n")
    return extraction_time

def demo_colpali():
    print("--- Starting ColPali ---")
    model_name = "vidore/colpali-v1.2"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading ColPali on {device}...")
    processor = ColPaliProcessor.from_pretrained(model_name)
    model = ColPali.from_pretrained(model_name, torch_dtype=torch.bfloat16).to(device)
    
    image = Image.open(IMAGE_PATH).convert("RGB")
    
    start_index = time.time()
    batch_images = processor.process_images([image]).to(device)
    with torch.no_grad():
        image_embeddings = model(**batch_images)
    indexing_time = time.time() - start_index
    print(f"ColPali Image Indexing Time: {indexing_time:.4f} seconds")

    start_query = time.time()
    batch_queries = processor.process_queries([TEST_QUERY]).to(device)
    with torch.no_grad():
        query_embeddings = model(**batch_queries)

    scores = processor.score_multi_vector(query_embeddings, image_embeddings)
    query_time = time.time() - start_query
    
    print(f"ColPali Query & Scoring Time: {query_time:.4f} seconds")
    print(f"Match Score for query '{TEST_QUERY}': {scores[0][0].item():.4f}\n")
    
    return indexing_time, query_time

if __name__ == "__main__":
    try:
        paddle_time = demo_paddleocr()
        colpali_idx, colpali_q = demo_colpali()
        
        print("--- Summary ---")
        print(f"OCR Extraction vs VLM Indexing: {paddle_time:.2f}s vs {colpali_idx:.2f}s")
    except Exception as e:
        print(f"Error during execution: {e}")