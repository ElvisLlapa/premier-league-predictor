import sys

import numpy
import pandas
import sklearn
import streamlit

print("Setup test passed!")
print(f"Python: {sys.version.split()[0]}")
print(f"NumPy: {numpy.__version__}")
print(f"Pandas: {pandas.__version__}")
print(f"scikit-learn: {sklearn.__version__}")
print(f"Streamlit: {streamlit.__version__}")