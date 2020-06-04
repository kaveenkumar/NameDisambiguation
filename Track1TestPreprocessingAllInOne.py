import multiprocessing as mp
import psutil
import json
import pathlib
import json
import re
import unidecode
import pandas as pd
from textblob import TextBlob
from textblob import Word
import nltk.corpus
from nltk.stem import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
import textdistance as tD
import py_stringmatching as sm
from nltk.corpus.reader import WordNetError
from scipy import spatial
import nltk
from nltk.corpus import wordnet
import math


def setOfLists(list):
    setOfLists = set()
    for i in list:
        setOfLists.add(tuple(i))
    return setOfLists


# Char edit based measure
def normTemplateCharEditSimLists(list1, list2, func, default):
    if len(list1) > 0 and len(list2) > 0:
        simSum = 0
        maxStr1 = ""
        maxStr2 = ""
        for str1 in set(list1):
            maxSim = 0
            for str2 in set(list2):
                sim = func(str1, str2)
                if sim > maxSim:
                    maxSim = sim
                    maxStr1 = str1
                    maxStr2 = str2
            simSum += maxSim * math.sqrt(list1.count(maxStr1) / len(list1) * list2.count(maxStr2) / len(list2))
        for str2 in set(list2):
            maxSim = 0
            for str1 in set(list1):
                sim = func(str1, str2)
                if sim > maxSim:
                    maxSim = sim
                    maxStr1 = str1
                    maxStr2 = str2
            simSum += maxSim * math.sqrt(list1.count(maxStr1) / len(list1) * list2.count(maxStr2) / len(list2))
        return simSum / 2
    else:
        # Average for target 0 around:
        return default


# Char edit based measure
def normDamLevSimStrings(str1, str2):
    lenMax = max(len(str1), len(str2))
    if lenMax > 0:
        return (lenMax - tD.damerau_levenshtein(str1, str2)) / lenMax
    else:
        # Average for target 0 around:
        return 0.22


# Char edit based measure
def normDamLevSimLists(list1, list2):
    return normTemplateCharEditSimLists(list1, list2, normDamLevSimStrings, 0.24)


# Char edit based measure
def normJaroWinkSimStrings(str1, str2):
    if len(str1) > 0 and len(str2) > 0:
        return tD.jaro_winkler(str1, str2)
    else:
        # Average for target 0 around:
        return 0.52


# Char edit based measure
def normJaroWinkSimLists(list1, list2):
    return normTemplateCharEditSimLists(list1, list2, normJaroWinkSimStrings, 0.57)


# Term based measure
def normJacSimStrings(str1, str2):
    if len(str1) > 0 and len(str2) > 0:
        return tD.jaccard(str1, str2)
    else:
        # Average for target 0 around:
        return 0.43


# Term based measure
def normJacSimLists(list1, list2):
    if len(list1) > 0 and len(list2) > 0:
        intersection = len(list(set(list1).intersection(list2)))
        union = (len(list1) + len(list2)) - intersection
        return 2 * float(intersection) / union
    else:
        # Average for target 0 around:
        return 0.35


# Hybrid measure
def normMongeElkanSimStrings(str1, str2):
    if len(str1) > 0 and len(str2) > 0:
        fun = sm.MongeElkan()
        return fun.get_raw_score([str1], [str2])
    else:
        # Average for target 0 around:
        return 0.58


# Hybrid measure
def normMongeElkanSimLists(list1, list2):
    return normTemplateCharEditSimLists(list1, list2, normMongeElkanSimStrings, 0.53)


# Term based measure
def normCosSimStrings(str1, str2):
    if len(str1) > 0 and len(str2) > 0:
        list1 = str1.split()
        list2 = str2.split()
        return normCosSimLists(list1, list2)
    else:
        # Average for target 0 around:
        return 0.093


# Term based measure
def normCosSimLists(list1, list2):
    if len(list1) > 0 and len(list2) > 0:
        words = set(set(list1) | set(list2))
        counts1 = []
        counts2 = []
        for word in words:
            counts1.append(list1.count(word))
            counts2.append(list2.count(word))

        return 1 - spatial.distance.cosine(counts1, counts2)
    else:
        # Average for target 0 around:
        return 0.015


# Term based measure
def normDiceSim(str1, str2, n):
    if len(str1) > 0 and len(str2) > 0:
        ngram1 = nGram(str1, n)
        ngram2 = nGram(str2, n)
        ngramSet1 = setOfLists(ngram1)
        ngramSet2 = setOfLists(ngram2)
        len1 = len(ngram1)
        len2 = len(ngram2)
        sim = 0
        if len1 > 0 and len2 > 0:
            for group1 in ngramSet1:
                for group2 in ngramSet2:
                    if group1 in ngramSet2:
                        counts1 = ngram1.count(list(group1))
                        counts2 = ngram2.count(list(group2))
                        sim += counts1 * counts2
            return sim / (len1 * len2)
        else:
            # Average for target 0 around:
            return 0.035
    else:
        # Average for target 0 around:
        return 0.035


# Knowledge based measure
def normAvgWuPalmSim(str1, str2):
    if len(str1) > 0 and len(str2) > 0:
        sim = 0
        comps = 0
        list1 = str1.split()
        list2 = str2.split()

        len1 = len(list1)
        len2 = len(list2)

        for word1 in set(list1):
            freq1 = list1.count(word1) / len1
            for word2 in set(list2):
                freq2 = list2.count(word2) / len2
                try:
                    w1 = wordnet.synset(word1 + ".n.01")
                    w2 = wordnet.synset(word2 + ".n.01")
                    sim += w1.wup_similarity(w2) * freq1 * freq2
                except WordNetError:
                    sim += 0.5 * freq1 * freq2

        return sim
    else:
        # Average for target 0 around:
        return 0.42


# Numerical based measure
def semiNormYearSim(year1, year2):
    if year1 != "" and year1 > 1600 and year2 != "" and year2 > 1600:
        # In the track 1 training data, ~ 99 % of authors didn't have a higher spectrum than 20 years
        generation = 20
        return 1 - abs(year1 - year2) / generation
    else:
        # Average for target 0 around:
        return 0.28


# Simplify words in string by transforming into singular forms
def lemma(str):
    result = []
    for word in TextBlob(str).split():
        # Using two different libraries for ensuring large coverage
        lemmatised = Word(word).lemmatize()
        WordNetLemmatizer().lemmatize(lemmatised)
        result.append(lemmatised)
    return " ".join(result)


# Create a list of n-sized lists with neighboring words in string
def nGram(str, n):
    wordList = str.split()
    ngram = []
    for word in range(len(wordList) - n + 1):
        ngram.append(wordList[word:word + n])
    return ngram


# Reduce string length and increase string meaningfulness

def preprocessLongString(str):
    if (type(str) is str):
        # Remove punctuation and special chars (keep dashes)
        str = re.sub(r"[.?!,;:/\\–\'\"()\[\]{}&|´`*+~#_°^$€<>%]", "", str)

        # Replace non-Unicode chars
        str = unidecode.unidecode(str)

        # Lowercase all
        str = str.lower()

        # Plural -> Singular (lemmatisation)
        str = lemma(str)

        try:
            # Tweak str to pandas dataframe for following operations
            dict = json.loads('{"0":"' + str + '"}', strict=False)
            strDF = pd.DataFrame([dict])
            strDF = strDF["0"]

            # Remove stop words (e.g. "is", "a", "the")
            stop = nltk.corpus.stopwords.words("english")
            strDF = strDF.apply(lambda x: " ".join(x for x in x.split() if x not in stop))

            # Don't remove rare words (e.g. n=1) because in precise scientific wording, they can still be important

            # Remove single char words
            strDF = strDF.apply(lambda x: " ".join(x for x in x.split() if len(x) > 1))

            # Don't do spelling correction because it takes too long
            # It can be assumed that scientific papers are mostly error-free

            # Remove suffixes (stemming)
            st = PorterStemmer()
            strDF = strDF.apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

            # Reduce all whitespaces, tabs, etc. to single spaces
            " ".join(strDF.values[0].split())

            return strDF.values[0]
        except:
            return str
    else:
        return ""


processCounter = 0

# Constants
fileAuthor = "../../../data/sna_test_data/sna_test_author_raw.json"
filePub = "../../../data/sna_test_data/test_pub_sna.json"
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
with open(fileAuthor) as file:
    authorsDic = json.load(file)

with open(filePub) as file:
    papersDic = json.load(file)

# Fetch training paper IDs
trainPaperIDs = []
comparisonsCount = 0
for authorName in authorsDic:
    print(authorName)
    authorNamePaperIDs = []

    for paper in authorsDic[authorName]:
        authorNamePaperIDs.append(
            {keyAuthor: authorName, keyPaper: paper})
    trainPaperIDs.append(authorNamePaperIDs)
    localPaperCount = len(authorNamePaperIDs)
    comparisonsCount += int((localPaperCount * localPaperCount - localPaperCount) / 2)


coreCount = int(psutil.cpu_count() - 2)

pathlib.Path('test_track1').mkdir(parents=True, exist_ok=True)


def comparePapers(firstChunk, lastChunk, n):
    with open("test_track1/test_track1_file" + str(n) + ".txt", 'w') as file:
        authorNameCounter = 0
        for authorNamePapers in trainPaperIDs[firstChunk:lastChunk+1]:
            print("Process " + str(n) + "; Progress: " + str(authorNameCounter) + "/" + str(lastChunk - firstChunk))
            authorNameCounter += 1
            for paperDic1 in authorNamePapers:
                mainAuthorName = paperDic1[keyAuthor]
                paper1 = paperDic1[keyPaper]
                outputStr = ""
                for paperDic2 in authorNamePapers:
                    paper2 = paperDic2[keyPaper]

                    # Only half of all pairs are unique
                    if (paper1 > paper2):

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

        for i in range(0, coreCount):
            firstIndex = lastIndex + 1
            lastIndex = firstIndex + int((authorNameCount / coreCount))

            if (lastIndex > authorNameCount):
                lastIndex = authorNameCount - 1

            processCounter += 1
            print("Test track 1 process " + str(processCounter) + " started")
            process = mp.Process(target=comparePapers, args=(firstIndex, lastIndex, processCounter))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()


threading(processCounter)
