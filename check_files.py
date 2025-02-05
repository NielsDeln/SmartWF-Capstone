import os

def check_files_in_ranges(folder_path, w_range, s_range, other_params):
    """
    Checks if all files with specified ranges exist in the folder.

    Args:
        folder_path (str): Path to the folder containing the files.
        w_range (tuple): A range for the 'w' parameter (start, end, step).
        s_range (tuple): A range for the 's' parameter (start, end, step).
        other_params (str): Fixed parameters in the file name (e.g., '_0_c_in').

    Returns:
        list: Missing files, if any.
    """
    print("Checking...")
    missing_files = []
    count = 0
    # Generate the expected filenames
    for w in range(*w_range):
        for s in range(*s_range):
            filename = f"w{w/10:.4f}_s{s/100:.2f}{other_params}"
            count += 1
            if filename not in os.listdir(folder_path):
                missing_files.append(filename)
                count -= 1
    
    return missing_files, count

#file name format: "w15.0000_s1.00_0_ms_in"

# Replace with your own data -->
folder_path = r"C:\Users\niels\Downloads\Dataset\Must_Should_Dataset_rep_4"   # Replace with your folder path
w_range = (150, 251, 1)          # From w10.50 to w20.0 would be: (105, 201, 1)
s_range = (100, 275, 25)            # From s0.0 to s0.3 in steps of 0.03: (0, 33, 3)
other_params = "_4_ms_out.out"   # Fixed part of the filename
missing, count = check_files_in_ranges(folder_path, w_range, s_range, other_params)
if missing:
    print(f"There are {count} files found but {len(missing)} files missing :(")
    print("Missing files:")
    print("\n".join(missing))
else:
    print(f"All {count} files are present! :)")
