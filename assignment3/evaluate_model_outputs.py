import argparse
import json
import re
import itertools
import numpy as np
from typing import Optional, Union, List
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description='Extract and evaluate answers from model outputs')
    parser.add_argument('--input_jsonl', type=str, required=True, help='Path to input JSONL file containing model outputs')
    parser.add_argument('--output_jsonl', type=str, required=True, help='Path to output JSONL file with scores and extracted answers')
    return parser.parse_args()


def estimate_pass_at_k(
        num_samples: Union[int, List[int], np.ndarray],
        num_correct: Union[List[int], np.ndarray],
        k: int
) -> np.ndarray:
    """
    Estimates pass@k of each problem and returns them in an array.

    Args:
        num_samples: Number of samples generated per problem
        num_correct: Number of correct samples per problem
        k: The k value for pass@k metric

    Returns:
        Array of pass@k estimates for each problem (Np Array)
    """

    # Convert to numpy arrays
    num_correct = np.array(num_correct)

    # Handle both single int and array for num_samples
    if isinstance(num_samples, int):
        num_samples_arr = np.full(len(num_correct), num_samples)
    else:
        num_samples_arr = np.array(num_samples)

    def compute_single_pass_at_k(n, c, k):
        """Compute pass@k for a single problem.

        pass@k = 1 - C(n-c, k) / C(n, k)

        For numerical stability, compute the ratio as a product:
        C(n-c, k) / C(n, k) = product_{i=0}^{k-1} (n-c-i) / (n-i)
        """
        # Edge case: not enough wrong samples to pick k wrong ones
        if n - c < k:
            return 1.0

        # Edge case: no correct samples
        if c == 0:
            return 0.0

        # Compute ratio using product formulation for numerical stability
        # C(n-c, k) / C(n, k) = product_{i=0}^{k-1} (n-c-i) / (n-i)
        ratio = 1.0
        for i in range(k):
            ratio *= (n - c - i) / (n - i)

        return 1.0 - ratio

    # Compute pass@k for each problem
    pass_at_k_values = np.array([
        compute_single_pass_at_k(n, c, k)
        for n, c in zip(num_samples_arr, num_correct)
    ])

    return pass_at_k_values

def load_jsonl_data(jsonl_path: str):
    """Load and validate JSONL data from inference output."""
    try:
        data = []
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line.strip()))

        # Validate required fields
        if not data:
            raise ValueError("JSONL file is empty")

        if 'model_output' not in data[0] or 'gt' not in data[0]:
            raise ValueError(f"JSONL must contain 'model_output' and 'gt' fields")

        return data
    except Exception as e:
        raise Exception(f"Error loading JSONL file: {str(e)}")

def extract_solution(solution_str: str) -> Optional[str]:
    """Extract the answer from the solution string.

    Extract the answer from the solution string in the format of \\boxed{answer}.
    If the answer is not in the format of \\boxed{answer}, return the last number in the solution string.
    Args:
        solution_str: the solution text

    Returns:
        The extracted answer as a string, or None if no answer found
    """

    if not solution_str:
        return None

    # Strategy 1: Try to extract from \boxed{...} format
    # Handle both \\boxed and \boxed patterns
    boxed_pattern = r'\\boxed\{([^}]+)\}'
    boxed_matches = re.findall(boxed_pattern, solution_str)
    if boxed_matches:
        # Take the last boxed answer (most likely the final answer)
        extracted = boxed_matches[-1].strip()
        # Clean up: remove commas, dollar signs, spaces
        extracted = extracted.replace(',', '').replace('$', '').replace(' ', '')
        if extracted:
            return extracted

    # Strategy 2: Look for "The answer is X" pattern
    answer_pattern = r'[Tt]he answer is[:\s]*([+-]?\d[\d,]*\.?\d*)'
    answer_matches = re.findall(answer_pattern, solution_str)
    if answer_matches:
        extracted = answer_matches[-1].replace(',', '')
        return extracted

    # Strategy 3: Fall back to extracting the last number in the text
    # Match integers, decimals, negative numbers (with optional commas)
    number_pattern = r'([+-]?\d[\d,]*\.?\d*)'
    numbers = re.findall(number_pattern, solution_str)
    if numbers:
        # Take the last number found
        extracted = numbers[-1].replace(',', '')
        return extracted

    return None


def compute_score(solution_str: str, ground_truth: str) -> int:
    """The scoring function for GSM8k.

    Compare both the extracted answer and the ground truth as float numbers.
    The score is 1 if the extracted answer is equal to the ground truth, 0 otherwise.
    """
    answer = extract_solution(solution_str)

    # Step 1: Check if answer was extracted
    if answer is None:
        return 0

    try:
        # Step 2: Normalize both answers by converting to float
        # This handles cases like "42" vs "42.0" vs "42.00"
        normalized_answer = str(float(answer))
        normalized_gt = str(float(ground_truth))

        # Step 3: Compare normalized values
        if normalized_answer == normalized_gt:
            return 1
        else:
            return 0

    except (ValueError, TypeError):
        # Handle non-numeric values
        return 0

def process_answers(data: list) -> tuple:
    """Process each answer and compute scores using simple extraction and comparison.

    Returns:
        tuple: (results list, grouped_by_idx dict)
    """
    results = []
    grouped_by_idx = defaultdict(list)

    correct_count = 0
    total_count = 0

    for item in data:
        extracted_answer_str = None
        grade = 0

        try:
            # Extract the answer from model output
            extracted_answer_str = extract_solution(item['model_output'])

            # Compute the score
            grade = compute_score(
                solution_str=item['model_output'],
                ground_truth=item['gt']
            )

            total_count += 1
            if grade == 1:
                correct_count += 1

        except Exception as e:
            print(f"Error processing item: {str(e)}")
            extracted_answer_str = None
            grade = 0

        # Create result with all original fields plus new ones
        result = item.copy()
        result['score'] = grade
        result['extracted_answer'] = extracted_answer_str

        results.append(result)

        # Group by idx if available (for multiple rollouts)
        idx = item.get('idx', total_count - 1)  # Use sequential index if idx not present
        grouped_by_idx[idx].append(result)

    # Calculate accuracy
    accuracy = correct_count / total_count if total_count > 0 else 0

    print(f"\nBasic Evaluation Results:")
    print(f"Total examples: {total_count}")
    print(f"Correct answers: {correct_count}")
    print(f"Overall Accuracy: {accuracy:.2%}")

    return results, grouped_by_idx


def compute_rollout_metrics(grouped_by_idx: dict, max_k: int = None) -> dict:
    """Compute avg@k and pass@k metrics for multiple rollouts.

    Args:
        grouped_by_idx: Dictionary mapping question idx to list of results
        max_k: Maximum k value to compute metrics for. If None, uses max rollouts available.

    Returns:
        Dictionary containing metrics for different k values
    """
    # Determine the number of rollouts
    rollout_counts = [len(rollouts) for rollouts in grouped_by_idx.values()]
    min_rollouts = min(rollout_counts)
    max_rollouts = max(rollout_counts)

    if max_k is None:
        max_k = max_rollouts

    # Check if all questions have the same number of rollouts
    all_same = min_rollouts == max_rollouts

    if not all_same:
        print(f"\nWarning: Different questions have different numbers of rollouts (min: {min_rollouts}, max: {max_rollouts})")
        print(f"Will compute metrics up to k={min_rollouts} to ensure fair comparison")
        max_k = min(max_k, min_rollouts)

    metrics = {}

    # Compute metrics for k from 1 to max_k
    for k in range(1, max_k + 1):
        # avg@k: average accuracy when randomly selecting k rollouts
        # For each question, calculate accuracy over first k rollouts
        accuracies_at_k = []
        for idx, rollouts in grouped_by_idx.items():
            # Take first k rollouts
            k_rollouts = rollouts[:k]
            # Calculate average accuracy
            avg_acc = sum(r['score'] for r in k_rollouts) / len(k_rollouts)
            accuracies_at_k.append(avg_acc)

        avg_at_k = np.mean(accuracies_at_k)

        # pass@k: probability that at least one of k rollouts is correct
        num_correct_per_question = []
        for idx, rollouts in grouped_by_idx.items():
            # Count how many of the first k rollouts are correct
            k_rollouts = rollouts[:k]
            num_correct = sum(r['score'] for r in k_rollouts)
            num_correct_per_question.append(num_correct)

        # Use the estimate_pass_at_k function
        pass_at_k_values = estimate_pass_at_k(
            num_samples=k,
            num_correct=num_correct_per_question,
            k=k
        )
        pass_at_k = np.mean(pass_at_k_values)

        metrics[k] = {
            'avg@k': avg_at_k,
            'pass@k': pass_at_k
        }

    return metrics

def main():
    args = parse_args()

    # Load input JSONL
    print(f"Loading data from {args.input_jsonl}")
    input_data = load_jsonl_data(args.input_jsonl)
    print(f"Loaded {len(input_data)} examples")

    # Process answers and compute scores
    results, grouped_by_idx = process_answers(input_data)

    # Check if we have multiple rollouts (more than 1 result per unique idx)
    num_unique_questions = len(grouped_by_idx)
    num_total_results = len(results)

    if num_total_results > num_unique_questions:
        print(f"\nDetected multiple rollouts: {num_unique_questions} unique questions with {num_total_results} total rollouts")

        # Compute and display rollout metrics
        metrics = compute_rollout_metrics(grouped_by_idx)

        print(f"\n{'='*60}")
        print("Multiple Rollout Metrics:")
        print(f"{'='*60}")
        print(f"{'k':<5} {'avg@k':<15} {'pass@k':<15}")
        print(f"{'-'*60}")
        for k in sorted(metrics.keys()):
            avg_k = metrics[k]['avg@k']
            pass_k = metrics[k]['pass@k']
            print(f"{k:<5} {avg_k:<15.4f} {pass_k:<15.4f}")
        print(f"{'='*60}")

        # Save metrics to a separate JSON file
        metrics_file = args.output_jsonl.replace('.jsonl', '_metrics.json')
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"\nMetrics saved to {metrics_file}")

    # Save results to output JSONL
    with open(args.output_jsonl, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"\nResults saved to {args.output_jsonl}")

if __name__ == "__main__":
    main()


