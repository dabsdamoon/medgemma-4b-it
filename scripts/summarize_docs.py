import argparse
import sys
import os
from pathlib import Path
from tqdm import tqdm

# Add parent directory to path to allow importing client
sys.path.insert(0, str(Path(__file__).parent.parent))

from client import MedGemmaClient

def main():
    parser = argparse.ArgumentParser(description="Batch summarize markdown documents to Markdown output.")
    parser.add_argument("--input_dir", type=str, default="raw", help="Input directory containing .md files")
    parser.add_argument("--output_dir", type=str, default="summaries", help="Output directory for summaries")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of files to process")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max tokens for generation")
    
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"Error: Input directory '{input_dir}' does not exist.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all markdown files
    files = sorted(input_dir.glob("*.md"))
    
    if args.limit:
        files = files[:args.limit]
        
    print(f"Found {len(list(input_dir.glob('*.md')))} files. Processing {len(files)} files...")
    
    try:
        client = MedGemmaClient()
    except ValueError as e:
        print(f"Error initializing client: {e}")
        print("Please ensure RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID environment variables are set.")
        return

    # specific prompt to ensure markdown output
    system_prompt = (
        "You are an expert medical professional. "
        "Provide a clear, accurate, and concise summary of the following medical document. "
        "Focus on key findings, diagnoses, treatments, and recommendations. "
        "Format your response in Markdown."
    )

    for file_path in tqdm(files, desc="Summarizing", unit="doc"):
        try:
            # Construct output filename: summary_{filename}.md
            output_filename = f"summary_{file_path.stem}.md"
            output_path = output_dir / output_filename

            if output_path.exists():
                tqdm.write(f"{output_filename} is skipped because it already exists.")
                continue

            content = file_path.read_text()
            
            # Truncate input if necessary to avoid context length issues (approx 30k chars)
            if len(content) > 30000:
                content = content[:30000]
            
            # Using async to avoid timeouts on longer documents
            summary = client.process_async(
                text=content,
                max_tokens=args.max_tokens,
                mode="summarize",
                system_prompt=system_prompt
            )
            
            output_path.write_text(summary)
            # print(f"Saved: {output_path}")
            
        except Exception as e:
            print(f"Failed to process {file_path.name}: {e}")

    print("\nBatch processing complete.")

if __name__ == "__main__":
    main()
