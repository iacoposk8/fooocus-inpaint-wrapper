import os
import shutil

def find_replace(file_path, find_text, replace_text):
    with open(file_path, 'r') as file:
        content = file.read()

    new_content = content.replace(find_text.replace('\\', '\\\\'), replace_text)

    with open(file_path, 'w') as file:
        file.write(new_content)

def delete_pycache_folders(start_dir='.'):
    for root, dirs, files in os.walk(start_dir):
        for dir_name in dirs:
            if dir_name == '__pycache__':
                folder_path = os.path.join(root, dir_name)
                try:
                    shutil.rmtree(folder_path)
                except Exception as e:
                    print("Error:",folder_path)
                    pass

delete_pycache_folders()

try:
    os.remove("Fooocus/config.txt")
except FileNotFoundError:
    pass
try:
    os.remove("Fooocus/modules/config.txt")
except FileNotFoundError:
    pass

fooocus_dir = os.path.dirname(os.path.realpath(__file__)) + "\\Fooocus"
find_replace(fooocus_dir + "\\modules\\config.py", fooocus_dir + "\\config.txt", './config.txt')
find_replace(fooocus_dir + "\\modules\\config.py", fooocus_dir + "\\presets\\default.json", './presets/default.json')

print("Done")