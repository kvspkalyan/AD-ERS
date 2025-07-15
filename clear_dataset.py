import os
import shutil

# Paths to dataset folders
dataset_dirs = {
    "Train": "dataset/train",
    "Validation": "dataset/val",
    "Test": "dataset/test"
}

# Set this to False if you want to delete the class folders too
preserve_structure = True

def clear_folder(folder_path):
    if not os.path.exists(folder_path):
        print(f"[SKIPPED] Folder not found: {folder_path}")
        return

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_path = os.path.join(root, file)
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"[ERROR] Couldn't delete {file_path}: {e}")

        if not preserve_structure:
            for dir in dirs:
                dir_path = os.path.join(root, dir)
                try:
                    shutil.rmtree(dir_path)
                except Exception as e:
                    print(f"[ERROR] Couldn't remove directory {dir_path}: {e}")

    print(f"[CLEANED] All files deleted from: {folder_path}")

def main():
    print("\n Starting Dataset Cleanup...\n")
    for name, path in dataset_dirs.items():
        print(f"→ Cleaning {name} dataset...")
        clear_folder(path)
    print("\n Dataset cleanup complete!\n")

if __name__ == "__main__":
    main()
