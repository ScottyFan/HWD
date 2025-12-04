"""
Reproduce the evaluation results from the paper (Table in the image)
Evaluates multiple models on IAM Words and IAM Lines datasets
"""

from hwd.datasets import GeneratedDataset
from hwd.scores import HWDScore, FIDScore, BFIDScore, KIDScore
import pandas as pd

def evaluate_model(dataset_name, model_name, reference_name='reference'):
    """
    Evaluate a single model on a dataset
    """
    print(f"\nEvaluating {model_name} on {dataset_name}...")
    
    try:
        # Load datasets
        fakes = GeneratedDataset(f'{dataset_name}__{model_name}')
        reals = GeneratedDataset(f'{dataset_name}__{reference_name}')
        
        # Calculate scores
        print(f"  Computing FID...")
        fid = FIDScore(height=32)
        fid_score = fid(fakes, reals)
        
        print(f"  Computing BFID...")
        bfid = BFIDScore(height=32)
        bfid_score = bfid(fakes, reals)
        
        print(f"  Computing KID...")
        kid = KIDScore(height=32)
        kid_score = kid(fakes, reals) * 1000  # Multiply by 10^3 as mentioned in paper
        
        print(f"  Computing HWD...")
        hwd = HWDScore(height=32)
        hwd_score = hwd(fakes, reals)
        
        return {
            'FID': fid_score,
            'BFID': bfid_score,
            'KID': kid_score,
            'HWD': hwd_score
        }
    
    except Exception as e:
        print(f"  Error: {e}")
        return None


def main():
    # Models to evaluate (based on the paper)
    models = [
        'ts_gan',
        'higanplus',
        'hwt',
        'vatr',
        'vatr_pp',
        'one_dm',
        'diffusionpen',
        'emuru'
    ]
    
    # Datasets to evaluate
    datasets = ['iam_words', 'iam_lines']
    
    # Results storage
    all_results = {}
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Evaluating on {dataset.upper()}")
        print(f"{'='*60}")
        
        results = {}
        
        for model in models:
            scores = evaluate_model(dataset, model)
            if scores:
                results[model] = scores
        
        all_results[dataset] = results
        
        # Print results table for this dataset
        print(f"\n\n{dataset.upper()} Results:")
        print(f"{'='*70}")
        
        df = pd.DataFrame(results).T
        df.index = df.index.str.upper().str.replace('_', '-')
        
        # Format the dataframe
        df = df.round(2)
        print(df.to_string())
        print(f"{'='*70}\n")
    
    # Save results to CSV
    print("\n\nSaving results to CSV files...")
    for dataset, results in all_results.items():
        df = pd.DataFrame(results).T
        df.index = df.index.str.upper().str.replace('_', '-')
        df = df.round(2)
        filename = f'{dataset}_results.csv'
        df.to_csv(filename)
        print(f"Saved {filename}")
    
    print("\n✓ Evaluation complete!")
    print("\nNote: The KID values are multiplied by 10^3 as mentioned in the paper.")
    print("Results may have slight variations due to random sampling in some metrics.")


if __name__ == "__main__":
    main()