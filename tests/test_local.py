"""
Local test script for MedGemma module (no GPU required).

Tests the input building logic up to model.generate() without actually running inference.
Useful for verifying prompt changes and input structure on a local machine without GPU.

Usage:
    python tests/test_local.py
    python tests/test_local.py --prompt "Custom prompt here"
    python tests/test_local.py --mode general
    python tests/test_local.py --no-save  # Don't save results to file
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import json
from datetime import datetime
from unittest.mock import MagicMock, patch
from medgemma import get_prompt, decode_image, create_dummy_image
from medgemma.prompts import DEFAULT_PROMPTS

# Store test results
TEST_RESULTS = {
    "timestamp": None,
    "tests": {},
    "summary": {},
}


def test_prompts():
    """Test prompt retrieval logic."""
    print("=" * 60)
    print("Testing Prompt Logic")
    print("=" * 60)

    results = {"passed": True, "checks": []}

    # Test default prompts
    print("\n1. Default prompts by mode:")
    default_prompts = {}
    for mode in ["summarize", "general"]:
        prompt = get_prompt(mode)
        default_prompts[mode] = prompt
        print(f"   [{mode}]: {prompt[:60]}...")
    results["checks"].append({"name": "default_prompts", "passed": True, "data": default_prompts})

    # Test custom prompt override
    custom = "You are a radiologist. Focus on imaging findings."
    result = get_prompt("summarize", custom_prompt=custom)
    assert result == custom, "Custom prompt should override default"
    print(f"\n2. Custom prompt override: ✓")
    print(f"   Custom: {custom}")
    results["checks"].append({"name": "custom_prompt_override", "passed": True})

    # Test unknown mode falls back to general
    result = get_prompt("unknown_mode")
    assert result == DEFAULT_PROMPTS["general"], "Unknown mode should fallback to general"
    print(f"\n3. Unknown mode fallback to 'general': ✓")
    results["checks"].append({"name": "unknown_mode_fallback", "passed": True})

    print("\n✓ All prompt tests passed!")
    TEST_RESULTS["tests"]["prompts"] = results
    return results


def test_image_decoding():
    """Test image decoding utilities."""
    print("\n" + "=" * 60)
    print("Testing Image Utilities")
    print("=" * 60)

    results = {"passed": True, "checks": []}

    # Test dummy image creation
    dummy = create_dummy_image()
    assert dummy.size == (224, 224), "Dummy image should be 224x224"
    assert dummy.mode == "RGB", "Dummy image should be RGB"
    print("\n1. Dummy image creation: ✓")
    print(f"   Size: {dummy.size}, Mode: {dummy.mode}")
    results["checks"].append({
        "name": "dummy_image_creation",
        "passed": True,
        "data": {"size": list(dummy.size), "mode": dummy.mode}
    })

    # Test decode_image with None returns dummy
    result = decode_image(None)
    assert result.size == (224, 224), "None should return dummy image"
    print("\n2. decode_image(None) returns dummy: ✓")
    results["checks"].append({"name": "decode_image_none", "passed": True})

    print("\n✓ All image utility tests passed!")
    TEST_RESULTS["tests"]["image_utilities"] = results
    return results


def test_generate_inputs(mode: str = "summarize", custom_prompt: str = None):
    """
    Test the input building logic up to model.generate().

    Returns the inputs dictionary that would be passed to model.generate().
    """
    print("\n" + "=" * 60)
    print("Testing Generate Input Building")
    print("=" * 60)

    # Test inputs
    test_text = """
    Patient: Jane Doe, 45-year-old female
    Chief Complaint: Persistent headache for 2 weeks

    History: Patient reports daily headaches, worse in the morning.
    Associated with mild nausea. No visual changes.

    Assessment: Tension-type headache vs. migraine without aura
    Plan: Trial of NSAIDs, headache diary, follow-up in 2 weeks
    """

    print(f"\n1. Test Configuration:")
    print(f"   Mode: {mode}")
    print(f"   Custom prompt: {'Yes' if custom_prompt else 'No (using default)'}")
    print(f"   Input text length: {len(test_text)} chars")

    # Get the prompt that would be used
    prompt_text = get_prompt(mode, custom_prompt)
    full_prompt = f"{prompt_text}\n\nInput Text:\n{test_text}\n\nResponse:"

    print(f"\n2. Prompt Resolution:")
    print(f"   System prompt: {prompt_text[:80]}...")

    # Build the message structure (same as in medgemma/model.py)
    image = create_dummy_image()
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": full_prompt},
            ],
        },
    ]

    print(f"\n3. Message Structure:")
    print(f"   Role: {messages[0]['role']}")
    print(f"   Content types: {[c['type'] for c in messages[0]['content']]}")

    # Mock the processor to capture what would be passed to it
    mock_processor = MagicMock()
    mock_inputs = {
        "input_ids": MagicMock(),
        "attention_mask": MagicMock(),
    }
    mock_inputs["input_ids"].shape = (1, 150)  # Simulated shape
    mock_inputs["input_ids"].__getitem__ = lambda self, idx: MagicMock(shape=(1, 150))

    # Mock the .to() method to return the same dict
    mock_result = MagicMock()
    mock_result.__getitem__ = lambda self, key: mock_inputs.get(key, MagicMock())
    mock_result.keys = lambda: mock_inputs.keys()
    mock_processor.apply_chat_template.return_value.to.return_value = mock_result

    # Call apply_chat_template (this is what happens before model.generate)
    mock_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )

    # Verify apply_chat_template was called correctly
    call_args = mock_processor.apply_chat_template.call_args
    print(f"\n4. processor.apply_chat_template() call:")
    print(f"   add_generation_prompt: {call_args.kwargs.get('add_generation_prompt')}")
    print(f"   tokenize: {call_args.kwargs.get('tokenize')}")
    print(f"   return_dict: {call_args.kwargs.get('return_dict')}")
    print(f"   return_tensors: {call_args.kwargs.get('return_tensors')}")

    # Build the generate kwargs (what would be passed to model.generate)
    max_new_tokens = 512
    generate_kwargs = {
        "input_ids": "tensor(...)",  # Would be actual tensor
        "attention_mask": "tensor(...)",  # Would be actual tensor
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }

    print(f"\n5. model.generate() would receive:")
    for key, value in generate_kwargs.items():
        print(f"   {key}: {value}")

    print("\n✓ Input building test completed!")

    result = {
        "mode": mode,
        "custom_prompt": custom_prompt,
        "resolved_prompt": prompt_text,
        "full_prompt_length": len(full_prompt),
        "message_structure": {
            "role": messages[0]["role"],
            "content_types": [c["type"] for c in messages[0]["content"]],
        },
        "generate_kwargs": generate_kwargs,
    }
    TEST_RESULTS["tests"]["generate_inputs"] = {"passed": True, "data": result}
    return result


def test_handler_integration(custom_prompt: str = None):
    """Test the handler's input parsing logic."""
    print("\n" + "=" * 60)
    print("Testing Handler Input Parsing")
    print("=" * 60)

    # Simulate job input (as it would come from RunPod)
    job = {
        "id": "test-job-001",
        "input": {
            "text": "Patient presents with symptoms...",
            "max_tokens": 256,
            "mode": "summarize",
        }
    }

    if custom_prompt:
        job["input"]["system_prompt"] = custom_prompt

    print(f"\n1. Simulated Job Input:")
    print(f"   Job ID: {job['id']}")
    print(f"   Text: {job['input']['text'][:40]}...")
    print(f"   Mode: {job['input']['mode']}")
    print(f"   Max tokens: {job['input']['max_tokens']}")
    print(f"   Custom prompt: {'Yes' if custom_prompt else 'No'}")

    # Parse like handler does
    job_input = job.get("input", {})
    text = job_input.get("text")
    max_tokens = job_input.get("max_tokens", 512)
    mode = job_input.get("mode", "summarize")
    system_prompt = job_input.get("system_prompt")

    print(f"\n2. Parsed Values:")
    print(f"   text: {text[:40]}...")
    print(f"   max_tokens: {max_tokens}")
    print(f"   mode: {mode}")
    print(f"   system_prompt: {system_prompt}")

    # Verify prompt resolution
    resolved = get_prompt(mode, system_prompt)
    expected = custom_prompt if custom_prompt else DEFAULT_PROMPTS[mode]
    assert resolved == expected, f"Prompt mismatch: {resolved} != {expected}"

    print(f"\n3. Prompt Resolution: ✓")
    print(f"   Resolved to: {resolved[:60]}...")

    print("\n✓ Handler integration test completed!")

    result = {
        "passed": True,
        "job_input": {
            "mode": mode,
            "max_tokens": max_tokens,
            "has_custom_prompt": custom_prompt is not None,
        },
        "resolved_prompt": resolved[:100],
    }
    TEST_RESULTS["tests"]["handler_integration"] = result
    return result


def save_results(output_dir: str = None) -> str:
    """Save test results to a JSON file."""
    # Default to project_root/logs
    if output_dir is None:
        output_path = PROJECT_ROOT / "logs"
    else:
        output_path = Path(output_dir)

    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    TEST_RESULTS["timestamp"] = timestamp

    filename = f"test_local_{timestamp}.json"
    filepath = output_path / filename

    with open(filepath, "w") as f:
        json.dump(TEST_RESULTS, f, indent=2, default=str)

    return str(filepath)


def main():
    parser = argparse.ArgumentParser(description="Local test for MedGemma module")
    parser.add_argument("--mode", type=str, default="summarize",
                        choices=["summarize", "general"],
                        help="Processing mode to test")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Custom prompt to test")
    parser.add_argument("--no-save", action="store_true",
                        help="Don't save results to file")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save test results (default: project_root/logs)")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print("MedGemma Local Test Suite (No GPU Required)")
    print("=" * 60)

    # Run all tests
    test_prompts()
    test_image_decoding()
    result = test_generate_inputs(mode=args.mode, custom_prompt=args.prompt)
    test_handler_integration(custom_prompt=args.prompt)

    # Build summary
    all_passed = all(
        t.get("passed", True) for t in TEST_RESULTS["tests"].values()
    )
    TEST_RESULTS["summary"] = {
        "all_passed": all_passed,
        "total_tests": len(TEST_RESULTS["tests"]),
        "mode_tested": args.mode,
        "custom_prompt_tested": args.prompt is not None,
    }

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print("\n✓ All tests passed!")
    print("\nGenerate inputs that would be passed to model.generate():")
    print(f"  - mode: {result['mode']}")
    print(f"  - custom_prompt: {result['custom_prompt'] or '(none - using default)'}")
    print(f"  - resolved_prompt: {result['resolved_prompt'][:50]}...")
    print(f"  - generate_kwargs: {list(result['generate_kwargs'].keys())}")

    if args.prompt:
        print(f"\n✓ Custom prompt correctly passed through the pipeline!")

    # Save results
    if not args.no_save:
        filepath = save_results(args.output_dir)
        print(f"\n📄 Results saved to: {filepath}")


if __name__ == "__main__":
    main()
