import pandas as pd
import json
import multiprocessing as mp
import psutil

fileTrainAuthor = "../../../data/train/train_author.json"
fileInputOutput = "training_track1/training_track1_file"
results = []

with open(fileTrainAuthor, "r") as file2:
    authorsDic = json.load(file2)

def correct(processNo):
    file  =fileInputOutput+ str(processNo)+".txt"
    with open(file, "r") as file1:
        counter = 0
        for line in file1:
            elements = line.strip().split(',')

            paper1 = str(elements[18])
            paper2 = str(elements[19])
            breakCond = False
            for key in authorsDic:
                subDic = authorsDic[key]
                paper1Found = False
                paper2Found = False
                for subKey in subDic:
                    subSubDic = subDic[subKey]
                    if paper1 in subSubDic:
                        paper1Found = True
                    if paper2 in subSubDic:
                        paper2Found = True
                    if paper1Found and paper2Found:
                        breakCond = True
                        elements[17] = key
                        break
                if(breakCond):
                    break
            results.append(elements)
            if(counter%10000 == 0):
                print("Process "+str(processNo)+": "+str(counter))
            counter += 1

    outputFrame = pd.DataFrame(results)
    outputFrame.to_csv(file, header=None, index=None, sep=',', mode='w')

coreCount = int(psutil.cpu_count() - 2)

def threading():
    if __name__ == '__main__':
        processes = []
        lastIndex = -1
        processCounter = 0

        for i in range(0,coreCount):
            processCounter += 1
            print("Process "+str(processCounter)+" started")
            process = mp.Process(target=correct, args=(processCounter,))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

threading()












