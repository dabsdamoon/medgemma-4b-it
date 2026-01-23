"""
Batch test script for summarizing markdown documents via RunPod.

Usage:
    python tests/test_batch.py /path/to/markdown/folder
    python tests/test_batch.py /path/to/markdown/folder --limit 5
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from client import MedGemmaClient


def main():
    parser = argparse.ArgumentParser(description="Batch summarize markdown documents")
    parser.add_argument("folder", type=str, help="Folder containing markdown files")
    parser.add_argument("--limit", type=int, default=3, help="Max files to process (default: 3)")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Max tokens per summary")
    parser.add_argument("--output", type=str, help="Output folder for summaries")
    args = parser.parse_args()

    folder = Path(args.folder)
    if not folder.exists():
        print(f"Error: Folder not found: {folder}")
        sys.exit(1)

    md_files = sorted(folder.glob("*.md"))[:args.limit]
    print(f"Found {len(list(folder.glob('*.md')))} markdown files, processing {len(md_files)}")
    print("=" * 60)

    try:
        client = MedGemmaClient()
    except ValueError as e:
        print(f"Error: {e}")
        print("\nSet environment variables:")
        print("  export RUNPOD_API_KEY=your_api_key")
        print("  export RUNPOD_ENDPOINT_ID=your_endpoint_id")
        sys.exit(1)

    output_folder = Path(args.output) if args.output else None
    if output_folder:
        output_folder.mkdir(parents=True, exist_ok=True)

    for i, md_file in enumerate(md_files, 1):
        print(f"\n[{i}/{len(md_files)}] Processing: {md_file.name}")
        print("-" * 40)

        with open(md_file, "r") as f:
            content = f.read()

        # Truncate if too long (model context limit ~8K tokens ≈ 32K chars)
        if len(content) > 30000:
            print(f"  (Truncating from {len(content)} to 30000 chars)")
            content = content[:30000]

        try:
            print("  Sending to API...")
            summary = client.process_async(
                text=content,
                max_tokens=args.max_tokens,
                mode="summarize"
            )
            print(f"\n  Summary:\n  {summary[:500]}{'...' if len(summary) > 500 else ''}")

            if output_folder:
                out_file = output_folder / f"{md_file.stem}_summary.txt"
                with open(out_file, "w") as f:
                    f.write(summary)
                print(f"  Saved to: {out_file}")

        except Exception as e:
            print(f"  Error: {e}")

    print("\n" + "=" * 60)
    print("Batch processing complete!")


if __name__ == "__main__":
    main()
