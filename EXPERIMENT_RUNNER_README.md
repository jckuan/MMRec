# Multi-Modal Experiment Runner

## Overview

This toolset allows you to easily run experiments with different modality configurations (text only, audio only, both) without manually editing config files.

## Available Scripts

### 1. Python Script (Recommended)
**File**: `run_experiments.py`
- Cross-platform (Windows, Linux, macOS)
- Full featured with error handling
- Automatic config backup and restoration
- Detailed logging

## Quick Start

### Python Script

```bash
# Run all modalities (text, audio, both, both+FETTLE)
python run_experiments.py --model DRAGON --dataset Music4All

# Run with specific split strategy
python run_experiments.py --model DRAGON --dataset Music4All --split cs_temporal_02

# Run specific modalities only
python run_experiments.py --model DRAGON --dataset Music4All --modalities text audio

# Run with different split strategy
python run_experiments.py --model DRAGON --dataset Music4All --split v1_timestamp

# Run on specific GPU
python run_experiments.py --model GRCN --dataset Music4All --gpu 1

# Don't save models (faster for testing)
python run_experiments.py --model LATTICE --dataset Music4All --no-save-model
```

## How It Works

### Modality Configurations

The scripts automatically modify your dataset YAML config to enable/disable modalities:

| Modality | Vision Features | Text Features | FETTLE | Use Case |
|----------|----------------|---------------|--------|----------|
| **text** | ❌ Disabled | ✅ Enabled | ❌ No | Text-only recommendation |
| **audio** | ✅ Enabled | ❌ Disabled | ❌ No | Audio-only recommendation |
| **both** | ✅ Enabled | ✅ Enabled | ❌ No | Multi-modal without alignment |
| **both_fettle** | ✅ Enabled | ✅ Enabled | ✅ Yes | Multi-modal with FETTLE alignment |

### Data Split Strategies
modifies feature files in your dataset config:

**Original** (`Music4All.yaml` with split `cs_lowcount_10`):
```yaml
inter_file_name: 'cs_lowcount_10/clean_lowcount_coldstart_interactions.csv'
vision_feature_file: 'cs_lowcount_10/clean_audio_feat_mert.npy'
text_feature_file: 'cs_lowcount_10/clean_text_feat.npy'
```

**With `--split cs_temporal_02`**:
```yaml
inter_file_name: 'cs_temporal_02/clean_temporal_coldstart_interactions.csv'
vision_feature_file: 'cs_temporal_02/clean_audio_feat_mert.npy'
text_feature_file: 'cs_temporal_02/clean_text_feat.npy'
```

**Text Only**:
```yaml
inter_file_name: 'cs_temporal_02/clean_temporal_coldstart_interactions.csv'
# vision_feature_file removed
text_feature_file: 'cs_temporal_02/clean_text_feat.npy'
```

**Audio Only**:
```yaml
inter_file_name: 'cs_temporal_02/clean_temporal_coldstart_interactions.csv'
vision_feature_file: 'cs_temporal_02/clean_audio_feat_mert.npy'
# text_feature_file removed
```

**Both / Both+FETTLE**:
```yaml
inter_file_name: 'cs_temporal_02/clean_temporal_coldstart_interactions.csv'
vision_feature_file: 'cs_temporal_02/clean_audio_feat_mert.npy'
text_feature_file: 'cs_temporal_02/clean_text_feat.npy'
# FETTLE enabled via config_dict parameter dataset config:

**Original** (`Music4All.yaml`):
```yaml
vision_feature_file: 'cs_lowcount_10/clean_audio_feat_mert.npy'
text_feature_file: 'cs_lowcount_10/clean_text_feat.npy'
```

**Text Only**:
```yaml
# vision_feature_file: 'cs_lowcount_10/clean_audio_feat_mert.npy'
text_featurewith Different Split Strategies

```bash
# Test all splits with same model
for split in cs_lowcount_10 cs_temporal_02 v1_timestamp; do
    python run_experiments.py --model DRAGON --dataset Music4All --split $split
done
```

### Running Specific Combinations

```bash
# Only text and audio without FETTLE
python run_experiments.py --model DRAGON --dataset Music4All --modalities text audio

# Only both with and without FETTLE
python run_experiments.py --model DRAGON --dataset Music4All --modalities both both_fettle

# Single modality with specific split
python run_experiments.py --model DRAGON --dataset Music4All --split cs_temporal_02 --modalities text
### Running Multiple Models

```bash
# Bash/PowerShell loop
for model in DRAGON GRCN LATTICE FREEDOM; do
    python run_experiments.py --model $model --dataset Music4All --split cs_temporal_02
done
```

### Running Multiple Datasets

```bash
for dataset in Music4All baby sports; do
    python run_experiments.py --model DRAGON --dataset $dataset
done
```

### Complete Split Strategy Comparison

```bash
# Compare all split strategies
splits=("cs_lowcount_10" "cs_temporal_02" "v1_timestamp")
for split in "${splits[@]}"; do
    python run_experiments.py --model DRAGON --dataset Music4All --split $split
don
### Running Multiple Datasets

```bash
for dataset in Music4All baby sports; do
    python run_experiments.py --model DRAGON --dataset $dataset
done
```

### Batch Experiments with FETTLE

```bash
# Compare with/without FETTLE for each modality
python run_experiments.py --model DRAGON --dataset Music4All --modalities text
python run_experiments.py --model DRAGON --dataset Music4All --modalities text --use-fettle

python run_experiments.py --model DRAGON --dataset Music4All --modalities audio
python run_experiments.py --model DRAGON --dataset Music4All --modalities audio --use-fettle

python run_experiments.py --model DRAGON --dataset Music4All --modalities both
python run_experiments.py --model DRAGON --dataset Music4All --modalities both --use-fettle
```

## Options Reference

### Python Script Options

```
--model, -m          Model name (required)
                     Choices: DRAGON, GRCN, LATTICE, FREEDOM, VBPR

--dataset, -d        Dataset name (required)
                     Example: Music4All, baby, sports

--modalities         Which modalities to run (default: all)
                     Choices: text, audio, both
  split, -s          Data split strategy (optional)
                     Examples: cs_lowcount_10, cs_temporal_02, v1_timestamp
                     If not specified, uses config default

--modalities         Which modalities to run (default: text, audio, both, both_fettle)
                     Choices: text, audio, both, both_fettle
                     Can specify multiple: --modalities text audio

--gpu               GPU device ID (default: 0)esting)
```

### PowerShell Script Options

```
-Model              Model name (required)
-Dataset            Dataset name (required)
-Modalities         Array of modalities (default: @("text","audio","both"))
-GPU                GPU device ID (default: 0)
-UseFETTLE          Enable FETTLE (switch)
-NoSaveModel        Don't save models (switch)
```

## Output

### Console Output

```
================================================================================
EXPERIMENT PLAN
================================================================================
Model: DRAGON
Dataset: Music4All
Split Strategy: cs_temporal_02
Modalities: text, audio, both, both_fettle
GPU: 0
Save models: Yes
Total experiments: 4
================================================================================

Proceed with experiments? [Y/n]: y

================================================================================
Running: DRAGON on Music4All with Text Only
Split Strategy: cs_temporal_02
================================================================================

✓ Modified Music4All.yaml:
  - Split strategy: cs_temporal_02
  - Vision features: disabled
  - Text features: enabled
  - FETTLE: disabled

[Training logs...]

✓ Experiment completed in 1234.56 seconds
✓ Restored original Music4All.yaml

... [more experiments] ...

================================================================================
EXPERIMENT SUMMARY
================================================================================
✓ Text Only: Success
✓ Audio Only: Success
✓ Both Modalities: Success
✓ Both Modalities + FETTLE: Success
================================================================================
```

### Log Files

Each experiment creates its own log directory:
```
MMRec/
├── log/
│   ├── DRAGON_Music4All_[timestamp]/  # Both modalities
│   ├── DRAGON_Music4All_[timestamp]/  # Text only
│   └── DRAGON_Music4All_[timestamp]/  # Audio only
```

## Troubleshooting

### Config Not Restored

If the script crashes and doesn't restore the config:

```bash
# Manually restore from backup
cd src/configs/dataset
cp Music4All.yaml.backup Music4All.yaml
```

### Permission Denied (PowerShell)

```powershell
# Enable script execution (run as Administrator)
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### Import Errors

Make sure you're running from the MMRec root directory:
```bash
cd /path/to/MMRec
python run_experiments.py --model DRAGON --dataset Music4All
```

### YAML Parsing Errors

If you see YAML errors, restore from backup:
```bash
cd src/configs/dataset
cp Music4All.yaml.backup Music4All.yaml
```

## Alternative Approaches

If you prefer more control, you can also:

### 1. Manual Config Files

Create separate config files:
```
configs/dataset/
├── Music4All.yaml           # Both modalities (original)
├── Music4All_text.yaml      # Text only
└── Music4All_audio.yaml     # Audio only
```

Then run:
```bash
python src/main.py --model DRAGON --dataset Music4All_text
python src/main.py --model DRAGON --dataset Music4All_audio
python src/main.py --model DRAGON --dataset Music4All
```

### 2. Command-Line Config Override

Modify `main.py` to accept config overrides:
```python
parser.add_argument('--disable-vision', action='store_true')
parser.add_argument('--disable-text', action='store_true')
```

### 3. Environment Variables

Set environment variables before running:
```bash
export DISABLE_VISION=1
python src/main.py --model DRAGON --dataset Music4All
```

## Best Practices

1. **Always check backups exist** before running experiments
2. **Use `--no-save-model`** when testing to save disk space
3. **Run one modality first** to verify setup before running all
4. **Monitor GPU memory** when running back-to-back experiments
5. **Document hyperparameters** in experiment notes
4 default configurations (includes FETTLE comparison)
python run_experiments.py --model DRAGON --dataset Music4All
```

### Split Strategy Comparison

```bash
# Compare different data splits
python run_experiments.py --model DRAGON --dataset Music4All --split cs_lowcount_10
python run_experiments.py --model DRAGON --dataset Music4All --split cs_temporal_02
python run_experiments.py --model DRAGON --dataset Music4All --split v1_timestamp
```

### Quick Testing

```bash
# Test single modality without saving
python run_experiments.py --model DRAGON --dataset Music4All --modalities text --no-save-model

# Test with specific split
python run_experiments.py --model DRAGON --dataset Music4All --split cs_temporal_02 --modalities audio --no-save-model
```

### Production Run

```bash
# Full experiment with specific split on GPU 1
python run_experiments.py --model DRAGON --dataset Music4All --split cs_temporal_02

### Production Run

```bash
# Full experiment with FETTLE on specific GPU
python run_experiments.py --model DRAGON --dataset Music4All --use-fettle --gpu 1
```

---

**Status**: ✅ Ready to use

**Tested on**: Python 3.8+, PowerShell 5.1+, Windows 10/11
