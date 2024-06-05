import os

def get_all_subdirectories(base_dir):
    subdirectories = []
    for root, dirs, files in os.walk(base_dir):
        for dir in dirs:
            subdirectories.append(os.path.join(root, dir))
    return subdirectories

# Usage example
base_directory = './injection-dataset_student'
all_subdirectories = get_all_subdirectories(base_directory)

for subdir in all_subdirectories:
    print(subdir)
