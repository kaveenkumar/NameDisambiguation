import glob

read_files = glob.glob("*.txt")

with open("result.txt", "wb") as outfile:
    for f in read_files:
        with open(f, "rb") as infile:
            filedata = infile.read()
            filedata = filedata.replace(b"yu liu,,", b"yu liu,")
            filedata = filedata.replace(b"lukens, james e.", b"lukens james e.")
            filedata = filedata.replace(b"a. c. buchanan, iii", b"a. c. buchanan iii")
            filedata = filedata.replace(b"sheng, ping", b"sheng ping")
            filedata = filedata.replace(b",\n", b"\n")
            outfile.write(filedata)