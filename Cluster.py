import pandas as pd
import numpy as np
import itertools
import scipy.spatial.distance as ssd
from scipy.cluster.hierarchy import dendrogram, linkage,fcluster
from matplotlib import pyplot as plt
import json

fileTrainAuthor = "../../../data/train/train_author.json"

with open(fileTrainAuthor) as file:
    authorsDic = json.load(file)

mainAuthorNames = []
for authorName in authorsDic:
    mainAuthorNames.append(authorName)




with open("networkOutput.txt", "r") as file:
    input = pd.read_table(file, sep=",", header=None)

input[input.columns[1]] = input[input.columns[1]].str.replace(' ', '_')
authorNames = input[input.columns[1]].unique()
authorNames = list(set(authorNames)&set(mainAuthorNames))


# Convert to distances
input[input.columns[0]] = [1 - value for value in input[input.columns[0]]]


resultDic = {}
authorCount = len(authorNames)
authorCounter = 0


for authorName in authorNames:
    authorCounter += 1
    print(str(authorCounter)+"/"+str(authorCount))
    authorPapers = input[input[input.columns[1]] == authorName]
    authorPapers.columns = ["dist", "author", "id1", "id2"]

    authorPapers = authorPapers[["id1", "id2", "dist"]]
    authorPapers2 = authorPapers[["id2", "id1", "dist"]]
    authorPapers2.columns = ["id1", "id2", "dist"]
    authorPapers = authorPapers.append(authorPapers2)

    all_paper_ids = authorPapers["id1"].unique().tolist()

    all_paper_ids.extend(authorPapers["id2"].unique().tolist())
    all_paper_ids = list(set(map(str, all_paper_ids)))



    combs = [all_paper_ids, all_paper_ids]
    combsList = pd.DataFrame(data=list(itertools.product(*combs)))
    combsList.columns = ["id1", "id2"]


    # Set unknown pairs to distance 1

    combsList = combsList.merge(authorPapers, how='left').fillna(1)

    # Set pairs of same IDs to distance 0
    combsList.loc[combsList["id1"] == combsList["id2"], 'dist'] = 0
    combsList = combsList.drop_duplicates(["id1","id2"])
    combsList = combsList.reset_index(drop=True)

    data_piv = combsList.pivot("id1", "id2", "dist")
    dist_mat = (data_piv.as_matrix() + np.transpose(data_piv.as_matrix()))/2



    # convert the redundant n*n square matrix form into a condensed nC2 array
    distArray = ssd.squareform(dist_mat)

    tree = linkage(distArray, method='ward', metric='euclidean')
    clustering = fcluster(tree, 0.65, 'distance').tolist()



    paperIDClusters  = [[] for i in range(max(clustering))]
    for i in range(0,len(all_paper_ids)):
        paperID = all_paper_ids[i]
        clusterIndex = clustering[i]
        paperIDClusters[clusterIndex-1].append(paperID)

    resultDic.update({authorName:paperIDClusters})

with open('clusteredResults.json', 'w') as file:
    json.dump(resultDic, file, indent = 4)














    #labelList = range(1, len(all_paper_ids))
    #plt.figure(figsize=(10, 7))
    #dendrogram(tree,
    #           orientation='top',
    #           # labels=labelList,
    #           distance_sort='ascending',
    #           show_leaf_counts=True)
    #plt.show()
