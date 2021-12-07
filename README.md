# Welcome to ML 7641 (Fall 2021) - Project Group 20

Team Members:
1. Tusheet Sidharth Goli - tgoli3
2. Tejas Rajamadam Pradeep – tpradeep8
3. Tillson Thomas Galloway – tgalloway7
4. Jan Jendrusak – jjendrusak3
5. Jiaxuan Chen – jchen813

# Final Project Report

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

[1] Kaggle: [https://www.kaggle.com/ahmednour/website-phishing-data-set](https://www.kaggle.com/ahmednour/website-phishing-data-set)

[2] Science Direct: [https://www.sciencedirect.com/science/article/pii/S2352340920313202](https://www.sciencedirect.com/science/article/pii/S2352340920313202)

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

## Potential Results/Discussion (Project Proposal)

We hope to obtain an accurate model that can classify if a website is phishing or not. Furthermore, we hope to identify which of our models performs better and to analyze the cases in which each excels and fails.

Additionally, we could contribute to the interpretability of our model by characterizing adversarial examples that result in misclassification. We could then perform novel work by analyzing the effect of a monotonicity property (Romeo, 2018) on the Decision Tree classifier in an attempt to increase the difficulty of an adversarial attack.

Our work is directly useful in a real-world situation: it could be implemented in browsers to warn/block users or email clients to analyze incoming/outgoing mail for spam classification.

## Results and Analysis (Final Report)

### Supervised Learning

### 1. Random Forest

* We used random forest and we tested it with different max_depths from 5 to 50  and n_estimators from 10 to 70. We found that test scores such as accuracies go up as max_depths and n_estimators increase until they reach a certain height. There are very little or no changes starting from max_depths of 30 and n_estimators of 40. The final scores are listed below:
    * F1: 0.956
    * Accuracy: 0.969
    * Balanced accuracy: 0.966
    * MCC: 0.932
    * Precision: 0.956
    * Recall: 0.955
    * FPR: 0.023

<img src="images\rf_result.png" alt="rf_result.png" height="360" width="470">

### 2. Neural Net

* We used a simple neural network with 3 hidden (linear) layers with ReLU activation functions, batch normalization, and dropouts after each linear layer with 0.1 probability. We could see that the results were quite good but even with hyperparameter tuning, the neural network couldn’t achieve better results. The final test results are as follows:
    * F1: 0.956
    * Accuracy: 0.969
    * Balanced accuracy: 0.966
    * MCC: 0.932
    * Precision: 0.956
    * Recall: 0.955
    * FPR: 0.023

<img src="images\nn_result.png" alt="nn_result.png" height="360" width="470">

### 3. Decision Tree

* We used a single decision tree and evaluated the model performance with different max_depths ranging from 5 to 80 (in increments of 10). We additionally varied the search criteria, selecting entropy and gini as potential solutions. We ultimately found that the entropy criterion with a max depth of 20 was the best.
* We used grid search to optimize hyperparameters on the validation dataset, namely: 
    * Criterion (gini or entropy)
    * Max depth
* The best decision tree was the one with entropy criterion and max depth of 20: [Accuracy: 0.950, Balanced accuracy: 0.945, F1: 0.928, Precision: 0.926, Recall: 0.930]
* We retrained a decision tree with these hyperparameters on merged training and validation data and it yielded the following test results:
    * F1: 0.932
    * Accuracy: 0.953
    * Balanced accuracy: 0.948
    * MCC: 0.897
    * Precision: 0.936
    * Recall: 0.929
    * FPR: 0.034

<img src="images\dt_new_result.png" alt="dt_new_result.png" height="360" width="470">

* Reasons
    * The 3 most important features (with corresponding feature importance values) in our decision tree classifier were:
        * “qty_dollar_file”: 0.516
        * “time_domain_activation”: 0.117
        * “directory_length”: 0.076
    * “qty_dollar_file” corresponds to the number of occurrences of “$” in the file section of URL - since there is no clear convention for using “$” in URL this seems rather arbitrary.
    * “time_domain_activation” corresponds to domain activation time (in days) - this sounds like a reasonable feature to be helpful to determine whether a web page is or is not phishing.
    * “directory_length” corresponds to the length of the directory section of URL - this might sound reasonable since phishing websites might try to use nested folders to create longer and therefore less readable URL.

### 4. k-NN

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

<img src="images\knn_result.png" alt="knn_result.png" height="360" width="470">

* Reasons
    * Having k = 1 as the best hyperparameter might be explained by web pages being highly diverse but when there is one that is almost identical (i.e. the closest neighbor) they will likely share the class label (being or not being a phishing web page).

### 5. SVM

* We used the SVM model provided in the sklearn library (sklearn.SVC()). The library provides four different kernels for the SVC, linear, poly, RBF, and sigmoid. Through testing, we found that the RBF kernel gave the best results with gamma scaling turned on, which means a value of 1 / (n_features * X.var()) is used for the value of gamma instead of 1 / (n_features). Both linear and poly kernels gave terrible results, with accuracies of around 20%, whereas the sigmoid kernel gave accuracies of around 50%. The best result was observed with the RBF kernel. The final scores are listed below:
    * F1: 0.624
    * Accuracy: 0.762
    * Balanced accuracy: 0.717
    * MCC: 0.455
    * Precision: 0.686
    * Recall: 0.572
    * FPR: 0.138

<img src="images\svm_result.png" alt="svm_result.png" height="360" width="470">

* Reasons
    * SVM generally does not work too well for very large datasets. Our dataset is very large and has a good amount of noise, likely causing SVM to have bad results. Even after running the model with hyperparameter tuning the results were not very high (60% for PCA, 75% for non-PCA). Our data is very tabular and hence favors models like random forest or decision trees over a model like SVM. By using PCA our results became worse due to the lack of features to vectorize for SVM.

### Unsupervised Learning

### 1. k-NN - data with reduced dimensionality (PCA) - 2 features

* The best k-NN classifier was the one using a single neighbor (k = 1): Accuracy: 0.820, Balanced accuracy: 0.804, F1: 0.742, Precision: 0.736, Recall: 0.749
* We retrained a k-NN classifier with this hyperparameter on merged training and validation data and it yielded the following test results:
    * F1: 0.756
    * Accuracy: 0.831
    * Balanced accuracy: 0.814
    * MCC: 0.627
    * Precision: 0.753
    * Recall: 0.760
    * FPR: 0.132

<img src="images\knn_pca_result.png" alt="knn_pca_result.png" height="360" width="470">

* Reasons
    * Having worse results than with a full dataset might be explained by the nature of PCA which doesn’t take the target variable into account. Therefore we might have discarded some knowledge from features that were actually important for predicting the class label but didn’t contribute to the variance very much.

### 2. SVM - data with reduced dimensionality (PCA)

* We applied PCA and ran our dataset against the same above SVM model we used for supervised learning. The final scores are listed below:
    * F1: 0.619
    * Accuracy: 0.602
    * Balanced accuracy: 0.603
    * MCC: 0.204
    * Precision: 0.621
    * Recall: 0.618
    * FPR: 0.414

<img src="images\svm_pca_result.png" alt="svm_pca_result.png" height="360" width="470">

* Reasons
    * SVM generally does not work too well for very large datasets. Our dataset is very large and has a good amount of noise, likely causing SVM to have bad results. Even after running the model with hyperparameter tuning the results were not very high (60% for PCA, 75% for non-PCA). Our data is very tabular and hence favors models like random forest or decision trees over a model like SVM. By using PCA our results became worse due to the lack of features to vectorize for SVM.

## Model Robustness

We calculated the importance of features for each of our models. For the decision tree-based models, we evaluated this using the mean accumulation of impurity decrease for each tree.

We found that across the models, the length of the directory component of a URL (in example.com/this/is/a/directory/?userId=1, “/this/is/a/directory/” is the directory) was the highest indicator of phishing vs. benign. In particular, the probability of a URL being marked as spam increased as the directory length increased. This can be observed directly through a CDF plot of the directory length for each dataset: the tail of the phishing distribution is much longer than the benign distribution.

<img src="images\robustness1.png" alt="robustness1.png">

Although the directory length feature was useful in the detection, exploiting the model's high reliance on this feature comes at a low cost to an attacker.

To demonstrate this, we modified each sample of our training set by changing the directory length on each phishing datapoint to a randomly sampled directory length from the benign dataset while keeping the other features fixed. This is a realistic assumption because an attacker with ownership of a server and domain name could realistically tweak the directory features arbitrarily without affecting other features. We were able to reduce the effectiveness of each of our models with this technique. Our best-performing model, the Random Forest, experienced a reduction in the false-negative rate from .8% to 4%. This attack had a 2.99% success rate on the Decision Tree across the full testing dataset, where a success is defined as the mislabelling of a sample that was originally labeled correctly due to manipulation of the feature vector. After adding in three more features relating to the characters used in the directory (number of ‘$’, number of ‘/’, number of ‘_’), we were able to increase the rate of successful evasions on the Random Forest to 30%. The CDF plots for each of the four features used are shown below. Note that in each case, the phishing URLs have a longer tail and therefore a larger mean than the benign URLs. Of the four features, the directory length was the most effective attack vector. Interestingly, the k-NN and Neural Network models were the most robust to this attack (k-NN: 6.96% success rate with four features; Neural Network: 4.24% success rate with four features), a trade-off with their long training and evaluation times.

<img src="images\robustness2.png" alt="robustness2.png">

<img src="images\robustness3.png" alt="robustness3.png">

A potential explanation for these results is that modern phishing attacks are oftentimes transmitted through trusted sites, for example, through a Google Form that generates a long, unique hash as part of the directory. We are unable to prove this, however, as the dataset does not include the actual phishing URLs. Future feature sets could include a feature indicative of shared hosting services (e.g. Google Forms or github.io), such as a binary variable for whether a site offers shared hosting or a measure of the number of malicious URLs under a certain domain name.

## Conclusions

### Validation Results

Overall based on our above models and respective analysis, the best performing model is the Random Forest. The Decision Tree also performs very well. We believe this is because of the tabular character of our dataset.

<img src="images\validation_result.png" alt="validation_result.png">

## Video

Proposal Video Link: [https://www.youtube.com/watch?v=25_FMB6S8uM](https://www.youtube.com/watch?v=25_FMB6S8uM)

<p align="center">
<iframe width="560" height="315" src="https://www.youtube.com/embed/25_FMB6S8uM" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
</p>

Final Video Link: 

## References

[1] FBI (2020). Internet Crime Report 2020. Federal Bureau of Investigation. [https://www.ic3.gov/Media/PDF/AnnualReport/2020_IC3Report.pdf](https://www.ic3.gov/Media/PDF/AnnualReport/2020_IC3Report.pdf).

[2] Íncer Romeo, Íñigo, et al. “Adversarially Robust Malware Detection Using Monotonic Classification.” Proceedings of the Fourth ACM International Workshop on Security and Privacy Analytics, Association for Computing Machinery, 2018, pp. 54–63. ACM Digital Library, [https://doi.org/10.1145/3180445.3180449](https://doi.org/10.1145/3180445.3180449).

[3] Vrbančič, Grega, et al. “Datasets for Phishing Websites Detection.” Data in Brief, vol. 33, Dec. 2020, p. 106438. ScienceDirect, [https://doi.org/10.1016/j.dib.2020.106438](https://doi.org/10.1016/j.dib.2020.106438).

## Project Timeline (Project Proposal)

Main Goal:
To begin working soon and finish the project before the beginning of finals week (Dec 1)

Assignments Due: Midpoint Report (Nov 16), Final Report (Dec 7)

Milestones to Aim For:
The name listed on each task does not mean that person must do the task himself, rather it means he is in charge of getting other team members to complete the task.

Expertise Areas: ML Coding: (All members), Github Workflows: (Jan), Video Editing: (Josh)

Data Cleanup: October 20 (Tejas)

Feature Selection: October 25th (Tusheet)

Initial attempt at Unsupervised Learning: November 1st (Each team member shall attempt his own version of unsupervised learning and see who gets the best results) (Tillson)

Initial attempt at Supervised Learning: November 1st (Each team member shall attempt his own version of supervised learning and see who gets the best results) (Tejas and Tusheet)

Finalize Unsupervised Learning: November 5th (Josh)

Finish Midterm Report: November 14th (Jan)

Polish results from both Supervised and Unsupervised Learnings: (Jan and Tillson)

Final Report and Video: December 6th (Josh)
