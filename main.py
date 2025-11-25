import subprocess
import sys

subprocess.run([sys.executable, "scripts/1_data_cleaning.py"], check=True)
subprocess.run([sys.executable, "scripts/2_feature_engineering.py"], check=True)
subprocess.run([sys.executable, "scripts/3_dimensionality_reduction.py"], check=True)
subprocess.run([sys.executable, "scripts/4_clustering.py"], check=True)
