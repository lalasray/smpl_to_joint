import os

def delete_non_imusim_files_and_empty_dirs(directory):
    """
    Recursively deletes all files that do NOT end with 'imusim.npz',
    and then removes all empty folders.

    Args:
        directory (str): Path to the directory to clean.
    """
    # First: delete non-imusim files
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            if not filename.endswith('.pkl'):
                file_path = os.path.join(dirpath, filename)
                try:
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                except Exception as e:
                    print(f"Failed to delete {file_path}: {e}")

    # Second: remove empty folders (walk bottom-up)
    for dirpath, dirnames, filenames in os.walk(directory, topdown=False):
        if not dirnames and not filenames:
            try:
                os.rmdir(dirpath)
                print(f"Removed empty folder: {dirpath}")
            except Exception as e:
                print(f"Failed to remove folder {dirpath}: {e}")

# Run on your target directory
delete_non_imusim_files_and_empty_dirs('/home/lala/Documents/Data/VQIMU/MMFIT')
