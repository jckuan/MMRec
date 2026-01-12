#!/usr/bin/env python
# coding: utf-8
"""
Multi-Modal Experiment Runner
==============================
Run experiments with different modality configurations:
- Text only
- Audio (vision) only  
- Both modalities

Usage:
    cd src
    python run_experiments.py --model DRAGON --dataset Music4All
    python run_experiments.py --model DRAGON --dataset Music4All --modalities text audio
    python run_experiments.py --model DRAGON --dataset Music4All --modalities both --gpu 0
"""

import os
import sys
import argparse
import yaml
import shutil
from pathlib import Path
from datetime import datetime

from utils.quick_start import quick_start


class ModalityConfig:
    """Helper class to manage modality configurations"""
    
    MODALITY_CONFIGS = {
        'text': {
            'name': 'Text Only',
            'vision_feature_file': None,
            'text_feature_file': 'keep',
            'use_fettle': False,
        },
        'audio': {
            'name': 'Audio Only',
            'vision_feature_file': 'keep',
            'text_feature_file': None,
            'use_fettle': False,
        },
        'both': {
            'name': 'Both Modalities',
            'vision_feature_file': 'keep',
            'text_feature_file': 'keep',
            'use_fettle': False,
        },
        'both_fettle': {
            'name': 'Both Modalities + FETTLE',
            'vision_feature_file': 'keep',
            'text_feature_file': 'keep',
            'use_fettle': True,
        }
    }
    
    @classmethod
    def get_config(cls, modality):
        """Get configuration for a specific modality"""
        if modality not in cls.MODALITY_CONFIGS:
            raise ValueError(f"Invalid modality: {modality}. Choose from {list(cls.MODALITY_CONFIGS.keys())}")
        return cls.MODALITY_CONFIGS[modality]
    
    @classmethod
    def list_modalities(cls):
        """List all available modalities"""
        return list(cls.MODALITY_CONFIGS.keys())


def backup_config(config_path):
    """Create a backup of the original config file"""
    backup_path = config_path.with_suffix('.yaml.backup')
    if not backup_path.exists():
        shutil.copy2(config_path, backup_path)
        print(f"✓ Created backup: {backup_path}")
    return backup_path


def parse_split_name(split_strategy):
    """Parse split strategy to extract key components for file naming.
    
    Args:
        split_strategy: e.g., 'cs_lowcount_10', 'cs_temporal_02', 'v1_timestamp'
    
    Returns:
        tuple: (interaction_filename, feature_prefix)
        
    Examples:
        'cs_lowcount_10' -> ('clean_lowcount_coldstart_interactions.csv', 'clean')
        'cs_temporal_02' -> ('clean_temporal_coldstart_interactions.csv', 'clean')
        'v1_timestamp' -> ('clean_v1_timestamp_interactions.csv', 'clean')
    """
    # Common patterns
    if split_strategy.startswith('cs_lowcount'):
        interaction_name = 'clean_lowcount_coldstart_interactions.csv'
    elif split_strategy.startswith('cs_temporal'):
        interaction_name = 'clean_temporal_coldstart_interactions.csv'
    elif split_strategy.startswith('v1'):
        interaction_name = 'clean_v1_timestamp_interactions.csv'
    else:
        # Generic fallback
        interaction_name = f'clean_{split_strategy}_interactions.csv'
    
    feature_prefix = 'clean'
    return interaction_name, feature_prefix


def modify_dataset_config(dataset_path, modality_config, original_config, split_strategy=None):
    """
    Temporarily modify the dataset config with modality settings.
    
    Args:
        dataset_path: Path to dataset YAML file
        modality_config: Modality configuration dict
        original_config: Original config content to restore later
        split_strategy: Optional split strategy (e.g., 'cs_lowcount_10', 'v1_timestamp')
    """
    with open(dataset_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Store original values for restoration
    original_values = {
        'vision_feature_file': config['vision_feature_file'],
        'text_feature_file': config['text_feature_file'],
        'inter_file_name': config['inter_file_name']
    }
    
    # Update split strategy paths if specified
    if split_strategy:
        interaction_name, feature_prefix = parse_split_name(split_strategy)
        config['inter_file_name'] = f'{split_strategy}/{interaction_name}'
        
        # Update feature file paths to match split strategy
        if 'vision_feature_file' in config:
            config['vision_feature_file'] = f'{split_strategy}/{feature_prefix}_audio_feat_mert.npy'
        if 'text_feature_file' in config:
            config['text_feature_file'] = f'{split_strategy}/{feature_prefix}_text_feat.npy'
    
    # Apply modality configuration
    for key, value in modality_config.items():
        if key in ['vision_feature_file', 'text_feature_file']:
            if value is None:
                # Remove the key (comment out equivalent)
                if key in config:
                    del config[key]
            elif value == 'keep':
                # Keep current value (possibly updated by split_strategy above)
                pass
            else:
                # Set new value
                config[key] = value
    
    # Write modified config
    with open(dataset_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    return original_values


def restore_dataset_config(dataset_path, original_content):
    """Restore the original dataset config"""
    with open(dataset_path, 'w') as f:
        f.write(original_content)


def run_experiment(model, dataset, modality, split_strategy=None, gpu_id=0, save_model=True):
    """
    Run a single experiment with specified modality configuration.
    
    Args:
        model: Model name (e.g., 'DRAGON', 'GRCN')
        dataset: Dataset name (e.g., 'Music4All')
        modality: Modality type ('text', 'audio', 'both', 'both_fettle')
        split_strategy: Data split strategy (e.g., 'cs_lowcount_10', 'v1_timestamp')
        gpu_id: GPU device ID
        save_model: Whether to save trained model
    """
    modality_config = ModalityConfig.get_config(modality)
    modality_name = modality_config['name']
    use_fettle = modality_config['use_fettle']
    
    print(f"\n{'='*80}")
    print(f"Running: {model} on {dataset} with {modality_name}")
    if split_strategy:
        print(f"Split Strategy: {split_strategy}")
    print(f"{'='*80}\n")
    
    # Prepare dataset config path (we're already in src/)
    dataset_config_path = Path('configs') / 'dataset' / f'{dataset}.yaml'
    
    if not dataset_config_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {dataset_config_path}")
    
    # Read original config content
    with open(dataset_config_path, 'r') as f:
        original_content = f.read()
    
    try:
        # Backup original config (first time only)
        backup_config(dataset_config_path)
        
        # Modify config for this modality
        original_values = modify_dataset_config(
            dataset_config_path, 
            modality_config,
            original_content,
            split_strategy=split_strategy
        )
        
        print(f"✓ Modified {dataset}.yaml:")
        if split_strategy:
            print(f"  - Split strategy: {split_strategy}")
        print(f"  - Vision features: {'disabled' if modality_config['vision_feature_file'] is None else 'enabled'}")
        print(f"  - Text features: {'disabled' if modality_config['text_feature_file'] is None else 'enabled'}")
        print(f"  - FETTLE: {'enabled' if use_fettle else 'disabled'}")
        print()
        
        # Prepare config dict for main
        config_dict = {
            'gpu_id': gpu_id,
            'use_fettle': use_fettle,
        }
        
        # Run the experiment
        start_time = datetime.now()
        quick_start(model=model, dataset=dataset, config_dict=config_dict, save_model=save_model)
        end_time = datetime.now()
        
        duration = (end_time - start_time).total_seconds()
        print(f"\n✓ Experiment completed in {duration:.2f} seconds")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Experiment failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Always restore original config
        restore_dataset_config(dataset_config_path, original_content)
        print(f"✓ Restored original {dataset}.yaml\n")


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-modal recommendation experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument('--model', '-m', type=str, required=True,
                       help='Model name (e.g., DRAGON, GRCN, LATTICE, FREEDOM, VBPR)')
    parser.add_argument('--dataset', '-d', type=str, required=True,
                       help='Dataset name (e.g., Music4All)')
    parser.add_argument('--split', '-s', type=str, default=None,
                       help='Data split strategy (e.g., cs_lowcount_10, cs_temporal_02, v1_timestamp)')
    parser.add_argument('--modalities', nargs='+', 
                       choices=ModalityConfig.list_modalities(),
                       default=['text', 'audio', 'both', 'both_fettle'],
                       help='Which modality configurations to run (default: text, audio, both, both_fettle)')
    parser.add_argument('--gpu', type=int, default=0,
                       help='GPU device ID (default: 0)')
    parser.add_argument('--no-save-model', action='store_true',
                       help='Do not save trained models')
    
    args = parser.parse_args()
    
    # Print experiment plan
    print("\n" + "="*80)
    print(f"EXPERIMENT PLAN")
    print("="*80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    if args.split:
        print(f"Split Strategy: {args.split}")
    print(f"Modalities: {', '.join(args.modalities)}")
    print(f"GPU: {args.gpu}")
    print(f"Save models: {not args.no_save_model}")
    print(f"Total experiments: {len(args.modalities)}")
    print("="*80 + "\n")
    
    # Confirm before starting
    try:
        response = input("Proceed with experiments? [Y/n]: ").strip().lower()
        if response and response not in ['y', 'yes']:
            print("Aborted.")
            return
    except KeyboardInterrupt:
        print("\nAborted.")
        return
    
    # Run experiments for each modality
    results = {}
    for modality in args.modalities:
        success = run_experiment(
            model=args.model,
            dataset=args.dataset,
            modality=modality,
            split_strategy=args.split,
            gpu_id=args.gpu,
            save_model=not args.no_save_model
        )
        results[modality] = 'Success' if success else 'Failed'
    
    # Print summary
    print("\n" + "="*80)
    print("EXPERIMENT SUMMARY")
    print("="*80)
    for modality, status in results.items():
        status_symbol = "✓" if status == "Success" else "✗"
        print(f"{status_symbol} {ModalityConfig.get_config(modality)['name']}: {status}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
