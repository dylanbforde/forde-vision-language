

import datasets
import requests
from PIL import Image
import io
import numpy as np
from transformers import AutoTokenizer

# Define standard image size for the vision model
IMAGE_SIZE = (224, 224)
# Define max text length for the tokenizer
MAX_TEXT_LENGTH = 128

def process_image(image_url):
    """
    Fetches an image from a URL, processes it, and returns it as a NumPy array.
    """
    try:
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        
        image = Image.open(io.BytesIO(response.content))
        image = image.convert("RGB")
        image = image.resize(IMAGE_SIZE)
        
        # Normalize pixel values to [0, 1]
        return np.array(image, dtype=np.float32) / 255.0
    except Exception:
        # Return None if any error occurs during fetching or processing
        return None

def create_dataset(tokenizer, data_dir=None):
    """
    Creates and preprocesses the Conceptual Captions dataset.
    If data_dir is provided, loads from disk. Otherwise, streams from Hugging Face.

    Args:
        tokenizer: A tokenizer object for processing the text captions.
        data_dir: Optional path to a saved dataset directory.

    Returns:
        A Hugging Face Dataset object.
    """
    if data_dir:
        print(f"Loading dataset from {data_dir}...")
        try:
            # Check for shards
            import os
            shards = [d for d in os.listdir(data_dir) if d.startswith('shard_') and os.path.isdir(os.path.join(data_dir, d))]
            
            if shards:
                print(f"Found {len(shards)} shards. Loading and concatenating...")
                shard_datasets = []
                # Sort by shard index
                shards.sort(key=lambda x: int(x.split('_')[1]) if x.split('_')[1].isdigit() else 999999)
                
                for shard in shards:
                    shard_path = os.path.join(data_dir, shard)
                    shard_datasets.append(datasets.load_from_disk(shard_path))
                
                dataset = datasets.concatenate_datasets(shard_datasets)
                return dataset
            else:
                # Try loading as a single dataset
                dataset = datasets.load_from_disk(data_dir)
                return dataset
        except Exception as e:
            print(f"Failed to load from {data_dir}: {e}")
            print("Falling back to streaming mode...")

    dataset = datasets.load_dataset("conceptual_captions", streaming=True, split="train")
    print("Dataset loaded in streaming mode.")

    def preprocess_function(examples):
        # Process images
        examples["image"] = [process_image(url) for url in examples["image_url"]]
        
        # Process captions
        tokenized_captions = tokenizer(
            examples["caption"], 
            padding="max_length", 
            truncation=True, 
            max_length=MAX_TEXT_LENGTH,
            return_tensors="np"
        )
        examples["input_ids"] = tokenized_captions["input_ids"]
        examples["attention_mask"] = tokenized_captions["attention_mask"]
        
        return examples

    processed_dataset = dataset.map(preprocess_function, batched=True)

    # Filter out examples where image processing failed
    filtered_dataset = processed_dataset.filter(lambda example: example["image"] is not None)
    
    return filtered_dataset

if __name__ == '__main__':
    # This example demonstrates the full preprocessing pipeline.
    print("Initializing tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    streaming_dataset = create_dataset(tokenizer)
    
    print("\nFetching a single processed example from the streaming dataset...")
    example = next(iter(streaming_dataset))
    
    print("\nExample keys:", example.keys())
    print("Image shape:", example["image"].shape)
    print("Image dtype:", example["image"].dtype)
    print("Input IDs (sample):", example["input_ids"][:20])
    print("Attention Mask (sample):", example["attention_mask"][:20])
    print("Original Caption:", example["caption"])
