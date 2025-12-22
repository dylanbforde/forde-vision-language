import os
import argparse
import datasets
from transformers import AutoTokenizer
from tqdm.auto import tqdm
import multiprocessing
from functools import partial
import sys

# Add project root to path to import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from src.data.dataset import process_image, MAX_TEXT_LENGTH

def mount_drive():
    """Mounts Google Drive if running in Colab."""
    # Check if already mounted
    if os.path.exists('/content/drive/MyDrive'):
        print("Google Drive already mounted.")
        return '/content/drive/MyDrive'

    try:
        from google.colab import drive
        print("Mounting Google Drive...")
        drive.mount('/content/drive')
        return '/content/drive/MyDrive'
    except ImportError:
        print("Not running in Google Colab. Skipping Drive mount.")
        return None
    except Exception as e:
        print(f"Could not mount Drive automatically: {e}")
        print("If running in Colab terminal, please mount Drive in a notebook cell first:")
        print("from google.colab import drive; drive.mount('/content/drive')")
        # Fallback: assume it might be mounted or user wants to save locally in Colab
        if os.path.exists('/content/drive'):
             return '/content/drive/MyDrive'
        return None

def process_example(example, tokenizer):
    """
    Processes a single example: downloads image and tokenizes caption.
    Returns None if image download fails.
    """
    image = process_image(example['image_url'])
    if image is None:
        return None
    
    tokenized_caption = tokenizer(
        example['caption'],
        padding='max_length',
        truncation=True,
        max_length=MAX_TEXT_LENGTH,
        return_tensors='np'
    )
    
    return {
        'image': image,
        'input_ids': tokenized_caption['input_ids'][0],
        'attention_mask': tokenized_caption['attention_mask'][0],
        'caption': example['caption']
    }

def download_and_save(output_dir, num_proc=4, max_samples=None):
    """
    Downloads, processes, and saves the dataset.
    """
    print(f"Preparing to save dataset to: {output_dir}")
    
    # Load the dataset in streaming mode first
    dataset = datasets.load_dataset("conceptual_captions", streaming=True, split="train")
    
    if max_samples:
        print(f"Limiting to {max_samples} samples.")
        dataset = dataset.take(max_samples)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # We need to iterate and process because streaming datasets don't support map with multiprocessing efficiently
    # for network bound tasks in the same way as non-streaming for saving to disk directly via save_to_disk
    # However, we want to save it as a HuggingFace dataset.
    # The best approach for a massive dataset like this to be saved to disk is to use `Dataset.from_generator`
    
    def generator():
        # Use a pool to process images in parallel
        # Note: We can't easily use multiprocessing pool with the generator directly yielding 
        # inside the pool without careful management. 
        # A simpler approach for this script is to just map sequentially or use dataset.map if we weren't streaming.
        # Since we are streaming, let's just use the dataset.map with batched=False but we can't easily parallelize network calls 
        # inside map without it being slow or complex.
        
        # Actually, `map` on streaming datasets DOES support arbitrary python code.
        # But to speed it up we want parallel downloads.
        # Let's use a ThreadPoolExecutor for downloads as they are I/O bound.
        
        from concurrent.futures import ThreadPoolExecutor
        
        buffer = []
        buffer_size = 100 # Process in chunks
        
        iterator = iter(dataset)
        
        with ThreadPoolExecutor(max_workers=num_proc) as executor:
            while True:
                chunk = list(itertools.islice(iterator, buffer_size))
                if not chunk:
                    break
                
                # Submit tasks
                futures = [executor.submit(process_example, ex, tokenizer) for ex in chunk]
                
                for future in futures:
                    result = future.result()
                    if result is not None:
                        yield result

    import itertools
    
    # Create a new dataset from the generator
    # We need to define features to ensure correct types, especially for images
    features = datasets.Features({
        'image': datasets.Array3D(shape=(224, 224, 3), dtype='float32'),
        'input_ids': datasets.Sequence(datasets.Value('int32')),
        'attention_mask': datasets.Sequence(datasets.Value('int8')),
        'caption': datasets.Value('string')
    })

    print("Starting download and processing... this may take a while.")
    processed_dataset = datasets.Dataset.from_generator(generator, features=features)
    
    print("Saving to disk...")
    processed_dataset.save_to_disk(output_dir)
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process dataset to Drive")
    parser.add_argument("--output_dir", type=str, default="forde_dataset", help="Directory name to save dataset")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    
    args = parser.parse_args()
    
    drive_root = mount_drive()
    
    if drive_root:
        full_output_path = os.path.join(drive_root, args.output_dir)
    else:
        # If not in Colab, save locally or to specified path
        full_output_path = args.output_dir
        
    download_and_save(full_output_path, num_proc=args.num_proc, max_samples=args.max_samples)
