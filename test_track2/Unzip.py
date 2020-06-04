import zipfile
import os
import zipfile

files = os.listdir(os.getcwd())
for file in files:
    if ".zip" in file:
        zip = zipfile.ZipFile(file)
        zip.extractall()
zip = None

for file in files:
    if ".zip" in file:
        os.remove(file)

