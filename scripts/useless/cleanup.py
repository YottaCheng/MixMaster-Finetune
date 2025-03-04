import os
import shutil

# Define the directories to clean
data_dir = "../data"
train_output_dir = os.path.join(data_dir, "train_raw_data/class_name_1")
test_output_dir = os.path.join(data_dir, "test_raw_data/class_name_1")

def clean_directory(directory):
    """
    Deletes all files in the specified directory.
    
    Args:
        directory (str): Path to the directory to clean.
    """
    if os.path.exists(directory):
        for filename in os.listdir(directory):
            file_path = os.path.join(directory, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)  # Delete the file
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)  # Delete the subdirectory
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Failed to delete {file_path}. Reason: {e}")
    else:
        print(f"Directory does not exist: {directory}")

# Clean the training and test output directories
clean_directory(train_output_dir)
clean_directory(test_output_dir)

print("✅ Cleanup complete! All generated files have been deleted.")