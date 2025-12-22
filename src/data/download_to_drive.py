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

def download_and_save(output_dir, num_proc=4, max_samples=None, shard_size=5000):
    """
    Downloads, processes, and saves the dataset in shards to allow resuming.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print(f"Preparing to save dataset to: {output_dir}")
    
    # 1. Detect existing shards to resume
    existing_shards = [d for d in os.listdir(output_dir) if d.startswith('shard_') and os.path.isdir(os.path.join(output_dir, d))]
    
    # Extract indices
    shard_indices = []
    for s in existing_shards:
        try:
            idx = int(s.split('_')[1])
            shard_indices.append(idx)
        except ValueError:
            pass
            
    if shard_indices:
        last_shard_idx = max(shard_indices)
        next_shard_idx = last_shard_idx + 1
        samples_processed = next_shard_idx * shard_size
        print(f"Found {len(shard_indices)} existing shards. Resuming from shard {next_shard_idx} (skipping {samples_processed} samples).")
    else:
        next_shard_idx = 0
        samples_processed = 0
        print("No existing shards found. Starting from scratch.")

    # 2. Load dataset and skip processed
    dataset = datasets.load_dataset("conceptual_captions", streaming=True, split="train")
    
    if samples_processed > 0:
        dataset = dataset.skip(samples_processed)
        
    if max_samples:
        # Adjust max_samples based on what's left
        remaining_samples = max_samples - samples_processed
        if remaining_samples <= 0:
            print("Max samples reached with existing shards. Exiting.")
            return
        print(f"Limiting to {remaining_samples} more samples.")
        dataset = dataset.take(remaining_samples)
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    # 3. Generator that yields chunks
    def chunked_generator():
        from concurrent.futures import ThreadPoolExecutor
        import itertools
        
        iterator = iter(dataset)
        
        # We need to process in batches of shard_size
        # But we also want parallel downloads within that
        
        current_shard_buffer = []
        
        # Inner generator for parallel processing
        def process_batch(batch_size=100):
            with ThreadPoolExecutor(max_workers=num_proc) as executor:
                while True:
                    chunk = list(itertools.islice(iterator, batch_size))
                    if not chunk:
                        break
                    
                    futures = [executor.submit(process_example, ex, tokenizer) for ex in chunk]
                    
                    results = []
                    for future in futures:
                        res = future.result()
                        if res is not None:
                            results.append(res)
                    yield results
                    
                    # If we got less than batch_size, we are done
                    if len(chunk) < batch_size:
                        break

        # Consume the parallel processor
        for batch_results in process_batch():
            for res in batch_results:
                yield res

    # We need to manually control the sharding loop because Dataset.from_generator 
    # consumes the whole thing.
    # So we will create a generator that yields exactly one shard, save it, 
    # and then create a new generator for the next shard.
    
    # Actually, re-creating the generator/iterator might be tricky with streaming.
    # A better way: Iterate over the main generator and collect into a list, 
    # then create a dataset from that list and save.
    
    features = datasets.Features({
        'image': datasets.Array3D(shape=(224, 224, 3), dtype='float32'),
        'input_ids': datasets.Sequence(datasets.Value('int32')),
        'attention_mask': datasets.Sequence(datasets.Value('int8')),
        'caption': datasets.Value('string')
    })

    import shutil

    print("Starting download and processing...")
    
    current_shard_data = []
    shard_counter = next_shard_idx
    
    # Re-using the logic from before but just iterating
    gen = chunked_generator()
    
    try:
        for example in tqdm(gen, desc="Processing"):
            current_shard_data.append(example)
            
            if len(current_shard_data) >= shard_size:
                # Save shard
                print(f"Shard {shard_counter} collected. Converting to Arrow format...")
                shard_dataset = datasets.Dataset.from_list(current_shard_data, features=features)
                
                # Save to a temporary local directory first to avoid Drive I/O latency
                temp_shard_dir = f"/content/temp_shards/shard_{shard_counter}"
                if os.path.exists(temp_shard_dir):
                    shutil.rmtree(temp_shard_dir)
                
                print(f"Saving shard {shard_counter} to temporary local storage ({temp_shard_dir})...")
                shard_dataset.save_to_disk(temp_shard_dir)
                
                # Move to final destination
                final_shard_dir = os.path.join(output_dir, f"shard_{shard_counter}")
                print(f"Moving shard {shard_counter} to final destination: {final_shard_dir}...")
                
                if os.path.exists(final_shard_dir):
                    shutil.rmtree(final_shard_dir)
                shutil.copytree(temp_shard_dir, final_shard_dir)
                
                # Cleanup temp
                shutil.rmtree(temp_shard_dir)
                
                print(f"Shard {shard_counter} successfully saved.")
                shard_counter += 1
                current_shard_data = [] # Reset buffer
                
        # Save remaining data
        if current_shard_data:
            print(f"Final shard collected. Converting to Arrow format...")
            shard_dataset = datasets.Dataset.from_list(current_shard_data, features=features)
            
            temp_shard_dir = f"/content/temp_shards/shard_{shard_counter}"
            if os.path.exists(temp_shard_dir):
                shutil.rmtree(temp_shard_dir)
                
            print(f"Saving final shard {shard_counter} to temporary local storage...")
            shard_dataset.save_to_disk(temp_shard_dir)
            
            final_shard_dir = os.path.join(output_dir, f"shard_{shard_counter}")
            print(f"Moving final shard to {final_shard_dir}...")
            
            if os.path.exists(final_shard_dir):
                shutil.rmtree(final_shard_dir)
            shutil.copytree(temp_shard_dir, final_shard_dir)
            
            shutil.rmtree(temp_shard_dir)
            print("Done!")
            
    except KeyboardInterrupt:
        print("\nInterrupted! Saving current progress...")
        if current_shard_data:
             print("Converting partial shard...")
             shard_dataset = datasets.Dataset.from_list(current_shard_data, features=features)
             
             temp_shard_dir = f"/content/temp_shards/shard_{shard_counter}_partial"
             print(f"Saving partial shard to {temp_shard_dir}...")
             shard_dataset.save_to_disk(temp_shard_dir)
             
             final_shard_dir = os.path.join(output_dir, f"shard_{shard_counter}_partial")
             print(f"Moving partial shard to {final_shard_dir}...")
             if os.path.exists(final_shard_dir):
                shutil.rmtree(final_shard_dir)
             shutil.copytree(temp_shard_dir, final_shard_dir)
             shutil.rmtree(temp_shard_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process dataset to Drive")
    parser.add_argument("--output_dir", type=str, default="forde_dataset", help="Directory name to save dataset")
    parser.add_argument("--num_proc", type=int, default=8, help="Number of worker threads")
    parser.add_argument("--max_samples", type=int, default=None, help="Limit number of samples for testing")
    parser.add_argument("--shard_size", type=int, default=1000, help="Number of samples per shard")
    
    args = parser.parse_args()
    
    drive_root = mount_drive()
    
    if drive_root:
        full_output_path = os.path.join(drive_root, args.output_dir)
    else:
        # If not in Colab, save locally or to specified path
        full_output_path = args.output_dir
        
    download_and_save(full_output_path, num_proc=args.num_proc, max_samples=args.max_samples, shard_size=args.shard_size)
