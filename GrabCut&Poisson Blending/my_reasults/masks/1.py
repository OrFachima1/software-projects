from PIL import Image
import os

def convert_fake_bmp_to_real_bmp(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.bmp'):
            file_path = os.path.join(folder_path, filename)
            try:
                # Try to open the file as a PNG
                with Image.open(file_path) as img:
                    img = img.convert('RGB')  # Convert to RGB to ensure compatibility with BMP format
                    img.save(file_path, format='BMP')
                    print(f"Converted {filename} to real BMP format.")
            except IOError:
                print(f"File {filename} is not a valid PNG file.")

# Use the current directory
folder_path = os.getcwd()
convert_fake_bmp_to_real_bmp(folder_path)
