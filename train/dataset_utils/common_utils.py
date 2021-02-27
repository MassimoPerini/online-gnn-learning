import time, sys, os
import urllib
from urllib.request import urlretrieve
import zipfile, shutil

def urllretrieve_reporthook(count, block_size, total_size):

    global start_time_urllretrieve

    if count == 0:
        start_time_urllretrieve = time.time()
        return

    duration = time.time() - start_time_urllretrieve + 1

    progress_size = int(count * block_size)
    speed = int(progress_size / (1024 * duration))
    percent = min(float(count*block_size*100/total_size),100)

    sys.stdout.write("\rDataReader: Downloaded {:.2f}%, {:.2f} MB, {:.0f} KB/s, {:.0f} seconds passed".format(
                    percent, progress_size / (1024 * 1024), speed, duration))

    sys.stdout.flush()

def downloadFromURL(URL, folder_path, zip=True):
    file_name = "temp.zip"
    # If directory does not exist, create
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    print("Downloading: {}".format(URL))
    print("In folder: {}".format(folder_path + file_name))

    try:
        urlretrieve (URL, folder_path + file_name, reporthook=urllretrieve_reporthook)

    except urllib.request.URLError as urlerror:

        print("Unable to complete automatic download, network error")
        raise urlerror

    sys.stdout.write("\n")
    sys.stdout.flush()

    if zip:
        print("Extracting...")
        dataFile = zipfile.ZipFile(folder_path + file_name)
        dataFile.extractall(path=folder_path)
        dataFile.close()
        print("cleaning temporary files...: ", folder_path + file_name)
        os.remove(folder_path + file_name)
        shutil.rmtree(folder_path+"__MACOSX", ignore_errors=True)

