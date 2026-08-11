import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.extract_embeddings_multilayer import extract_layers

extract_layers("test")

print("Test multi-layer embedding extraction complete!")
