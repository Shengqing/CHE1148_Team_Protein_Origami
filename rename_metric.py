import re

files_to_update = [
    "src/eval.py",
    "src/train.py",
    "src/vae_train.py",
    "scripts/train_esm2_regressor.py",
    "scripts/train_graphnet_generative.py"
]

for file in files_to_update:
    with open(file, "r") as f:
        content = f.read()

    # Apply re substitutions
    content = re.sub(r'spearman', 'kendall', content)
    content = re.sub(r'Spearman', 'Kendall', content)
    content = re.sub(r'kendallr', 'kendalltau', content) # scipy.stats.kendalltau
    
    # We replaced `sp` to `kendall` in some places but not the var names, wait we just want to safely rename metric.
    
    with open(file, "w") as f:
        f.write(content)
