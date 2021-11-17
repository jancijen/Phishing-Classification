# Welcome to ML 7641 (Fall 2021) - Project Group 20

Team Members:
1. Tusheet Sidharth Goli - tgoli3
2. Tejas Rajamadam Pradeep – tpradeep8
3. Tillson Thomas Galloway – tgalloway7
4. Jan Jendrusak – jjendrusak3
5. Jiaxuan Chen – jchen813

# Midterm Project Report

## Project Name

Phishing Classification

## Introduction/Background
In 2020, phishing was the most prevalent type of cybercrime, costing the American public over $54 million in losses. While employee training against phishing has increased in past years, attack sophistication has increased alongside it. A common cyber kill chain involving phishing begins with an email, text, or phone call that directs the victim to a website to harvest credentials, credit card numbers, and sensitive information. Preemptive identification of a website as malicious would allow system administrators to block or warn clients against imminent cyber threats.

## Problem Definition

We used a recent dataset (Vrbančič, 2020) containing 88,647 labeled 111-dimensional features to predict whether a website is phishing or legitimate. 96 of the features in the dataset contain information extracted from the domain name and IP, and the other 15 contain information about the website itself. Each of the instances has been verified by multiple sources via the industry-standard PhishTank registry. 30,647 of the instances are labeled as phishing and the other 58,000 instances are labeled as legitimate.

## Methods

Unsupervised: We begin by running Principal Component Analysis on the feature set to reduce the dimensionality and to optimize our training.

Supervised: We train three models using the reduced feature set, namely a model based on decision trees, one based on neural networks, and one based on an SVM to predict whether unseen examples are phishing or legitimate.

Our entire pipeline can be implemented using scikit-learn and we can perform data visualizations using Seaborn and Matplotlib.

Approach Ideas: [https://towardsdatascience.com/phishing-domain-detection-with-ml-5be9c99293e5](https://towardsdatascience.com/phishing-domain-detection-with-ml-5be9c99293e5)

## Datasets

Kaggle: [https://www.kaggle.com/ahmednour/website-phishing-data-set](https://www.kaggle.com/ahmednour/website-phishing-data-set)

Science Direct: [https://www.sciencedirect.com/science/article/pii/S2352340920313202](https://www.sciencedirect.com/science/article/pii/S2352340920313202)

## Data

We have an imbalanced “full” dataset with 88647 data points and class ratio (phishing/non-phishing) of 0.528, and a “balanced” “small” subset with 58645 data points and class ratio (phishing/non-phishing) of 1.095. In the following sections will report results for the “full” dataset. The unbalanced dataset is more reflective of the real world: there are more benign emails than phishing emails on the internet.

<img src="images\data1.png" alt="data1.png">

Most of our features in the dataset are URL related according to the following URL sub-sections: (Vrbančič, 2020)

<img src="images\data2.png" alt="data2.png">

### Data Split

For the train-validation-test split we used a stratified split to split the dataset (88647 data points) as follows:
* 60% - training set (53188 data points)
* 20% - validation set (17729 data points)
* 20% - test set (17730 data points)

All of the splits have the same class ratio (phishing/non-phishing) of 0.528.

### Data Analysis

We ran t-SNE to reduce the full training dataset from 112 features to 2 and colored the data points according to their class (phishing or non-phishing website):

<img src="images\data3.png" alt="data3.png">

### Data Pre-Processing

Features Dimensionality Reduction
* We attempted to use feature reduction using PCA
    * We used PCA to have fewer dimensions only for our k-NN baseline model, because a decision tree, in some sort, performs “feature selection” by itself.
    * Our dataset had 111 features and we felt like using a feature reduction algorithm shall help make our results better.
* We used the in-built PCA function found in sklearn.decomposition to perform the features dimensionality reduction.
* For the actual PCA values, we tested a number of combinations of retained variances to try and see which ones gave us the best results and we ended up using 99% retained variance which resulted in 2 dimensions/features.
* What we found out tho was using PCA for dimensionality reduction actually did not improve the results of k-NN at all. It in fact made the results a little worse.
* The reason we think this is because the data points are quite different and have many parameters they differ by, so performing a k-NN algorithm did not lead to good clustering.

## Results and Analysis

We concentrated on Supervised Learning and implemented several different ways of performing Supervised Learning on out dataset.

### Decision Tree

1. Decision Tree
    * We used grid search to optimize hyperparameters on the validation dataset, namely: 
        * Criterion (gini or entropy)
        * Max depth
    * The best decision tree was the one with entropy criterion and max depth of 20: Accuracy: 0.950, Balanced accuracy: 0.945, F1: 0.928, Precision: 0.926, Recall: 0.930
    * We retrained a decision tree with these hyperparameters on merged training and validation data and it yielded the following test results:
        * F1: 0.932
        * Accuracy: 0.953
        * Balanced accuracy: 0.948
        * MCC: 0.897
        * Precision: 0.936
        * Recall: 0.929
        * FPR: 0.034

<img src="images\dt_result.png" alt="dt_result.png">

* Analysis
    * The 3 most important features (with corresponding feature importance values) in our decision tree classifier were:
        * “qty_dollar_file”: 0.516
        * “time_domain_activation”: 0.117
        * “directory_length”: 0.076
    * “qty_dollar_file” corresponds to the number of occurrences of “$” in the file section of URL - since there is no clear convention for using “$” in URL this seems rather arbitrary.
    * “time_domain_activation” corresponds to domain activation time (in days) - this sounds like a reasonable feature to be helpful to determine whether a web page is or is not phishing.
    * “directory_length” corresponds to the length of the directory section of URL - this might sound reasonable since phishing websites might try to use nested folders to create longer and therefore less readable URL.

### k-NN

2. k-NN
    * We used grid search to optimize hyperparameters on the validation dataset, namely: 
        * Number of neighbors
    * The best k-NN classifier was the one using a single neighbor (k = 1): Accuracy: 0.883, Balanced accuracy: 0.872, F1: 0.832, Precision: 0.827, Recall: 0.837
    * We retrained a k-NN classifier with this hyperparameter on merged training and validation data and it yielded the following test results:
        * F1: 0.846
        * Accuracy: 0.893
        * Balanced accuracy: 0.883
        * MCC: 0.764
        * Precision: 0.841
        * Recall: 0.851
        * FPR: 0.085

<img src="images\knn_result.png" alt="knn_result.png">

* Analysis
    * Having k = 1 as the best hyperparameter might be explained by web pages being highly diverse but when there is one that is almost identical (i.e. the closest neighbor) they will likely share the class label (being or not being a phishing web page).

### k-NN - data with reduced dimensionality (PCA) - 2 features

3. k-NN - data with reduced dimensionality (PCA) - 2 features
    * The best k-NN classifier was the one using a single neighbor (k = 1): Accuracy: 0.820, Balanced accuracy: 0.804, F1: 0.742, Precision: 0.736, Recall: 0.749
    * We retrained a k-NN classifier with this hyperparameter on merged training and validation data and it yielded the following test results:
        * F1: 0.756
        * Accuracy: 0.831
        * Balanced accuracy: 0.814
        * MCC: 0.627
        * Precision: 0.753
        * Recall: 0.760
        * FPR: 0.132

<img src="images\knn_pca_result.png" alt="knn_pca_result.png">

* Analysis
    * Having worse results than with a full dataset might be explained by the nature of PCA which doesn’t take the target variable into account. Therefore we might have discarded some knowledge from features that were actually important for predicting the class label but didn’t contribute to the variance very much.

## Potential Results/Discussion (Project Proposal)

We hope to obtain an accurate model that can classify if a website is phishing or not. Furthermore, we hope to identify which of our models performs better and to analyze the cases in which each excels and fails.

Additionally, we could contribute to the interpretability of our model by characterizing adversarial examples that result in misclassification. We could then perform novel work by analyzing the effect of a monotonicity property (Romeo, 2018) on the Decision Tree classifier in an attempt to increase the difficulty of an adversarial attack.

Our work is directly useful in a real-world situation: it could be implemented in browsers to warn/block users or email clients to analyze incoming/outgoing mail for spam classification.

## References

FBI (2020). Internet Crime Report 2020. Federal Bureau of Investigation. [https://www.ic3.gov/Media/PDF/AnnualReport/2020_IC3Report.pdf](https://www.ic3.gov/Media/PDF/AnnualReport/2020_IC3Report.pdf)

Vrbančič, Grega, et al. “Datasets for Phishing Websites Detection.” Data in Brief, vol. 33, Dec. 2020, p. 106438. ScienceDirect, [https://doi.org/10.1016/j.dib.2020.106438](https://doi.org/10.1016/j.dib.2020.106438).

Íncer Romeo, Íñigo, et al. “Adversarially Robust Malware Detection Using Monotonic Classification.” Proceedings of the Fourth ACM International Workshop on Security and Privacy Analytics, Association for Computing Machinery, 2018, pp. 54–63. ACM Digital Library, [https://doi.org/10.1145/3180445.3180449](https://doi.org/10.1145/3180445.3180449).

## Video

Link: [https://www.youtube.com/watch?v=25_FMB6S8uM](https://www.youtube.com/watch?v=25_FMB6S8uM)

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/25_FMB6S8uM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

## Timeline

Main Goal:
To begin working soon and finish the project before the beginning of finals week (Dec 1)

Assignments Due: Midpoint Report (Nov 16), Final Report (Dec 7)

Milestones to Aim For:
The name listed on each task does not mean that person must do the task himself, rather it means he is in charge of getting other team members to complete the task.

Expertise Areas: ML Coding: (All members), Github Workflows: (Jano), Video Editing: (Josh)

Data Cleanup: October 20 (Tejas)

Feature Selection: October 25th (Tusheet)

Initial attempt at Unsupervised Learning: November 1st (Each team member shall attempt his own version of unsupervised learning and see who gets the best results) (Tillson)

Initial attempt at Supervised Learning: November 1st (Each team member shall attempt his own version of supervised learning and see who gets the best results) (Tejas and Tusheet)

Finalize Unsupervised Learning: November 5th (Josh)

Finish Midterm Report: November 14th (Jano)

Polish results from both Supervised and Unsupervised Learnings: (Jano and Tillson)

Final Report and Video: December 5th (Josh)
