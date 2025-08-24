import os

print("--- Starting Path Diagnosis ---")

# --- Let's see the current working directory ---
current_working_dir = os.getcwd()
print(f"1. Current Working Directory:\n   {current_working_dir}\n")

# --- Let's build the path like in your script ---
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"2. The script's absolute directory is:\n   {script_dir}\n")

    # This is the path we are trying to build
    constructed_path = os.path.join(script_dir, "data", "Avenue Dataset")
    print(f"3. The constructed path to the dataset should be:\n   {constructed_path}\n")

    # --- Let's check if key parts exist ---
    data_folder_path = os.path.join(script_dir, "data")
    print(f"4. Checking if the 'data' folder exists...")
    if os.path.exists(data_folder_path):
        print("   ✅ SUCCESS: 'data' folder found.\n")
        
        # If 'data' exists, let's see what's inside it
        print(f"5. Contents of the 'data' folder:")
        try:
            contents = os.listdir(data_folder_path)
            if not contents:
                print("   ⚠️  The 'data' folder is empty.")
            else:
                for item in contents:
                    print(f"   - {item}")
            print("")

        except Exception as e:
            print(f"   ❌ ERROR: Could not read contents of 'data' folder: {e}\n")

        print(f"6. Finally, checking for the full constructed path...")
        if os.path.exists(constructed_path):
            print("   ✅ SUCCESS: The full path to 'Avenue Dataset' was found!")
        else:
            print("   ❌ FAILURE: The full path to 'Avenue Dataset' was NOT found.")
            print("   ❗️ACTION: Check for typos or case-sensitivity issues in your folder name.")
            print("             Based on the contents list above, is 'Avenue Dataset' spelled correctly?")

    else:
        print("   ❌ FAILURE: The 'data' folder was NOT found in the script's directory.")
        print("   ❗️ACTION: Make sure your 'data' folder is in the same directory as the script.")

except Exception as e:
    print(f"An unexpected error occurred: {e}")

print("\n--- Diagnosis Complete ---")