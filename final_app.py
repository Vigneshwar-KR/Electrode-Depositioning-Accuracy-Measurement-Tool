
import os
from utils import *


def main():
    Tk().withdraw()  
    folder_path = filedialog.askdirectory(title="Select the directory containing the images")

    if folder_path:
        process_images(folder_path)
    else:
        print("No folder selected.")

if __name__ == "__main__":
    main()
