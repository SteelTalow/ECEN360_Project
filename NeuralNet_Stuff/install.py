# Step 1: Check Python Version and Environment
import sys

print(f"✅ Your Python version: {sys.version}")

# Ensure compatibility
if sys.version_info >= (3, 8):
    print("🎉 Your Python version is compatible!")
else:
    print("❌ Your Python version is lower than 3.8. Please upgrade Python to version 3.8 or higher.")
    print("Follow this guide: https://realpython.com/installing-python/")





import subprocess

# List of required libraries
required_libraries = ["pandas", "google-cloud-bigquery", "google-cloud-storage", "google-auth", "google-cloud-bigquery","pyarrow","db_dtypes","matplotlib",
                      "scikit-learn", "ISLP", "torchinfo","torchvision"]

# Function to check and install missing libraries
def check_and_install_library(library):
    try:
        __import__(library.split("-")[0])  # Check if the library is installed
        print(f"✅ Library '{library}' is already installed.")
    except ImportError:
        print(f"❌ Library '{library}' is missing.")
        print(f"Installing '{library}'...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", library])
            print(f"🎉 Successfully installed '{library}'!")
        except Exception as e:
            print(f"🚨 Error installing '{library}': {e}")
            print("👉 Try running the following command in your terminal:")
            print(f"pip install {library}")

# Check and install libraries
for lib in required_libraries:
    check_and_install_library(lib)
