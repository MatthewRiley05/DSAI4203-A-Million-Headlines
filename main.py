import subprocess
import sys

subprocess.run([sys.executable, "scripts/1_data_cleaning.py"])
subprocess.run([sys.executable, "scripts/2_feature_engineering.py"])
subprocess.run([sys.executable, "scripts/3_dimensionality_reduction.py"])
subprocess.run([sys.executable, "scripts/4_clustering.py"])
