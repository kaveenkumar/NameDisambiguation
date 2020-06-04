# NameDisambiguation
Author Name Disambiguation by Clustering based on Deep Learned Pairwise Similarities

Course: 6CP project in Applied Deep Learning
Collaborator: Philipp Gnoyke

## Abstract
Name disambiguation in the field of scientific literature management is a rising issue. Authors might share the same name as other authors and distinguishing who is who is a challenging task. This report describes an intent to solve this problem by applying deep learning. We extracted a series of pairwise similarity measures from bibliographic metadata that were used for training and predicting with neural networks. The trained model predicts a likelihood of two papers belonging to the same person. We used the likelihoods to assign papers into existing clusters and to construct author profiles from scratch with agglomerative clustering. For pairwise classification, an F1 of 51.5 % was reached. For clustering from scratch and assigning into existing clusters, we reached clustering F1s of 48 % and 84.3 % respectively.
