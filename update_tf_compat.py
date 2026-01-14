#!/usr/bin/env python3
import os
import re

files_to_update = [
    'extern/dnnlib/fused_bias_act_n.py',
    'extern/dnnlib/upfirdn_2d_n.py',
    'extern/metrics/FID.py',
    'models/ops.py',
    'models/loss.py',
    'util/util.py',
    'util/image_util.py',
    'train.py'
]

for file_path in files_to_update:
    full_path = os.path.join('/home/someone/iOrthopredictor', file_path)
    if not os.path.exists(full_path):
        print(f"Skipping {file_path} - not found")
        continue

    with open(full_path, 'r') as f:
        content = f.read()

    # Check if already updated
    if 'tensorflow.compat.v1' in content:
        print(f"Skipping {file_path} - already updated")
        continue

    # Replace import tensorflow as tf
    content = re.sub(
        r'^import tensorflow as tf$',
        'import tensorflow.compat.v1 as tf\ntf.disable_v2_behavior()',
        content,
        flags=re.MULTILINE
    )

    with open(full_path, 'w') as f:
        f.write(content)

    print(f"Updated {file_path}")

print("Done!")
