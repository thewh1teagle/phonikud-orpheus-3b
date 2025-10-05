"""
wget https://huggingface.co/datasets/thewh1teagle/saspeech/resolve/main/saspeech_manual/saspeech_manual_v2.7z
sudo apt install p7zip-full -y
7z x saspeech_manual_v2.7z

uv run src/prepare_data.py
"""

import pandas as pd

# id, text, phonemes
df = pd.read_csv("saspeech_manual/metadata.csv",sep="\t",header=None)
# id|phonemes (LJSpeech format)
df[[0,2]].to_csv("saspeech_manual/metadata_phonemes.csv",index=False, sep="|", header=None)