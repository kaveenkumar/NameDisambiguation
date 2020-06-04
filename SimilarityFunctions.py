import textdistance as tD
import py_stringmatching as sm
from nltk.corpus.reader import WordNetError
from scipy import spatial
import nltk
from nltk.corpus import wordnet
from TextMining import nGram
from Utils import setOfLists
import math

# Uncomment when running for the first time!
# nltk.download()

# Char edit based measure
def normTemplateCharEditSimLists(list1,list2,func, default):
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
    return normTemplateCharEditSimLists(list1,list2,normDamLevSimStrings,0.24)


# Char edit based measure
def normJaroWinkSimStrings(str1, str2):
    if len(str1) > 0 and len(str2) > 0:
        return tD.jaro_winkler(str1, str2)
    else:
        # Average for target 0 around:
        return 0.52


# Char edit based measure
def normJaroWinkSimLists(list1, list2):
    return normTemplateCharEditSimLists(list1,list2,normJaroWinkSimStrings,0.57)


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
    return normTemplateCharEditSimLists(list1,list2,normMongeElkanSimStrings,0.53)


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
