from SimilarityFunctions import *
from TextMining import *
import multiprocessing as mp
import psutil
import json
from os import path


processCounter = 0

# Constants
fileTrainAuthor = "../../../data/train/train_author.json"
fileTrainPub = "../../../data/train/train_pub.json"
keyAuthorProfile = "author_profile"
keyPaper = "paper"
keyAuthor = "author"
keyAuthors = "authors"
keyName = "name"
keyOrg = "org"
keyTitle = "title"
keyAbstract = "abstract"
keyKeywords = "keywords"
keyVenue = "venue"
keyYear = "year"

# Load data from files
with open(fileTrainAuthor) as file:
    authorsDic = json.load(file)

with open(fileTrainPub) as file:
    papersDic = json.load(file)




# Fetch training paper IDs
trainPaperIDs = []
comparisonsCount = 0
for authorName in authorsDic:
    authorNamePaperIDs = []
    for authorProfile in authorsDic[authorName]:
        for paper in authorsDic[authorName][authorProfile]:
            authorNamePaperIDs.append(
                {keyAuthorProfile: authorProfile, keyAuthor: authorName,
                 keyPaper: paper})
    trainPaperIDs.append(authorNamePaperIDs)
    localPaperCount = len(authorNamePaperIDs)
    comparisonsCount += int((localPaperCount * localPaperCount - localPaperCount) / 2)

# print(comparisonsCount)

coreCount = int(psutil.cpu_count() - 2)

def comparePapers(firstChunk,lastChunk,n):
    posCounter = 0
    negCounter = 0
    fileName = "training_track1/training_track1_file"+str(n)+".txt"
    if(path.exists(fileName)):
        with open(fileName, 'r') as file:
            try:
                existentPapers = pd.read_table(file, sep=",", header=None)
                existentPaperPairs = existentPapers[existentPapers.columns[18:20]].values.tolist()
                existentPaperTargets = existentPapers[existentPapers.columns[16]]

                posCounter = existentPaperTargets.sum()
                negCounter = len(existentPaperTargets) - posCounter

            except:
                existentPaperPairs = []
    else:
        with open(fileName, 'w') as file:
            existentPaperPairs = []

    with open("training_track1/training_track1_file"+str(n)+".txt", 'a') as file:
        authorNameCounter = 0
        for authorNamePapers in trainPaperIDs[firstChunk:lastChunk]:
            print("Process "+str(n)+"; Progress: "+str(authorNameCounter)+"/"+str(lastChunk-firstChunk))
            authorNameCounter += 1
            for paperDic1 in authorNamePapers:
                mainAuthorName = paperDic1[keyAuthor]
                paper1 = paperDic1[keyPaper]
                authorProfile1 = paperDic1[keyAuthorProfile]
                outputStr = ""
                for paperDic2 in authorNamePapers:
                    paper2 = paperDic2[keyPaper]
                    authorProfile2 = paperDic2[keyAuthorProfile]

                    tuple1 = [paper1, paper2]
                    tuple2 = [paper2, paper1]

                    alreadyThere = tuple1 in existentPaperPairs or tuple2 in existentPaperPairs

                    if(not alreadyThere):
                        if authorProfile1 == authorProfile2:
                            targetClass = 1
                        else:
                            targetClass = 0

                        if(negCounter > 10000 and posCounter > 10000):
                            posVsNeg = posCounter / negCounter
                        else:
                            posVsNeg = 1

                        if(posVsNeg > 0.95 and targetClass == 0 or posVsNeg < 1.05 and targetClass == 1):
                            proceed = True
                        else:
                            proceed = False

                        # Only half of all pairs are unique
                        if (proceed and paper1 > paper2):
                            authorProfile2 = paperDic2[keyAuthorProfile]

                            if targetClass == 1:
                                posCounter += 1
                            else:
                                negCounter += 1

                            coAuthorsList1 = []
                            coOrgsList1 = []
                            coAuthorsList2 = []
                            coOrgsList2 = []
                            org1 = ""
                            org2 = ""

                            # Load attributes from paper 1
                            if keyAuthors in papersDic[paper1]:
                                authorsList1 = papersDic[paper1][keyAuthors]
                                if len(authorsList1) > 0:
                                    for authorDic in authorsList1:
                                        authorName = authorDic[keyName].lower()
                                        if keyOrg in authorDic:
                                            org = authorDic[keyOrg].lower()
                                        else:
                                            org = ""
                                        if authorName != paperDic1[keyAuthor].replace("_", " ").lower():
                                            coAuthorsList1.append(authorName)
                                            coOrgsList1.append(org)
                                        else:
                                            org1 = org
                            else:
                                authorsList1 = []
                                coAuthorsList1 = []

                            if keyTitle in papersDic[paper1]:
                                title1 = preprocessLongString(papersDic[paper1][keyTitle])
                            else:
                                title1 = ""

                            if keyAbstract in papersDic[paper1]:
                                abstract1 = preprocessLongString(papersDic[paper1][keyAbstract])
                            else:
                                abstract1 = ""

                            if keyKeywords in papersDic[paper1]:
                                keywordsList1 = papersDic[paper1][keyKeywords]
                            else:
                                keywordsList1 = []

                            if keyVenue in papersDic[paper1]:
                                venue1 = papersDic[paper1][keyVenue]
                            else:
                                venue1 = ""

                            if keyYear in papersDic[paper1]:
                                year1 = papersDic[paper1][keyYear]
                            else:
                                year1 = 0

                            # Load attributes from paper 2
                            if keyAuthors in papersDic[paper2]:
                                authorsList2 = papersDic[paper2][keyAuthors]
                                if len(authorsList2) > 0:
                                    for authorDic in authorsList2:
                                        authorName = authorDic[keyName].lower()
                                        if keyOrg in authorDic:
                                            org = authorDic[keyOrg].lower()
                                        else:
                                            org = ""
                                        if authorName != paperDic2[keyAuthor].replace("_", " ").lower():
                                            coAuthorsList2.append(authorName)
                                            coOrgsList2.append(org)
                                        else:
                                            org2 = org
                            else:
                                coAuthorsList2 = []

                            if keyTitle in papersDic[paper2]:
                                title2 = preprocessLongString(papersDic[paper2][keyTitle])
                            else:
                                title2 = ""

                            if keyAbstract in papersDic[paper2]:
                                abstract2 = preprocessLongString(papersDic[paper2][keyAbstract])
                            else:
                                abstract2 = ""

                            if keyKeywords in papersDic[paper2]:
                                keywordsList2 = papersDic[paper2][keyKeywords]
                            else:
                                keywordsList2 = []

                            if keyVenue in papersDic[paper2]:
                                venue2 = papersDic[paper2][keyVenue]
                            else:
                                venue2 = ""

                            if keyYear in papersDic[paper2]:
                                year2 = papersDic[paper2][keyYear]
                            else:
                                year2 = 0

                            # Compute similarities
                            outputStr += str(normCosSimLists(coAuthorsList1, coAuthorsList2))
                            outputStr += "," + str(normDamLevSimLists(coAuthorsList1, coAuthorsList2))

                            outputStr += "," + str(normCosSimLists(coOrgsList1, coOrgsList2))
                            outputStr += "," + str(normDamLevSimLists(coOrgsList1, coOrgsList2))

                            outputStr += "," + str(normCosSimStrings(org1, org2))
                            outputStr += "," + str(normDamLevSimStrings(org1, org2))

                            outputStr += "," + str(normCosSimStrings(venue1, venue2))
                            outputStr += "," + str(normDamLevSimStrings(venue1, venue2))

                            outputStr += "," + str(normCosSimLists(keywordsList1, keywordsList2))
                            outputStr += "," + str(normDamLevSimLists(keywordsList1, keywordsList2))

                            outputStr += "," + str(normDiceSim(preprocessLongString(" ".join(keywordsList1)),
                                                               preprocessLongString(" ".join(keywordsList2)), 2))

                            outputStr += "," + str(normCosSimStrings(title1, title2))
                            outputStr += "," + str(normDiceSim(title1, title2, 2))

                            outputStr += "," + str(normCosSimStrings(abstract1, abstract2))
                            outputStr += "," + str(normDiceSim(abstract1, abstract2, 2))

                            outputStr += "," + str(semiNormYearSim(year1, year2))

                            outputStr += "," + str(targetClass)

                            outputStr += "," + mainAuthorName.replace(",", "_")
                            outputStr += "," + paper1.replace(",", "")
                            outputStr += "," + paper2.replace(",", "") + "\n"

                            outputStr = unidecode.unidecode(outputStr)

                file.write(outputStr)



def threading(processCounter):
    if __name__ == '__main__':
        processes = []
        authorNameCount = len(trainPaperIDs)

        lastIndex = -1

        for i in range(0,coreCount):
            firstIndex = lastIndex+1
            lastIndex = firstIndex + int((authorNameCount/coreCount))

            if(lastIndex>authorNameCount):
                lastIndex = authorNameCount-1

            processCounter += 1
            print("Train process "+str(processCounter)+" started")
            process = mp.Process(target=comparePapers, args=(firstIndex,lastIndex,processCounter))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()

threading(processCounter)


