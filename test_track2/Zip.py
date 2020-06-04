import zipfile
try:
    import zlib
    mode= zipfile.ZIP_DEFLATED
except:
    mode= zipfile.ZIP_STORED
import os

files = os.listdir(os.getcwd())
for file in files:
    if ".py" not in file and ".zip" not in file:
        zipfile.ZipFile(file+'.zip', 'w',mode).write(file)

for file in files:
    if ".py" not in file and ".zip" not in file:
        os.remove(file)
