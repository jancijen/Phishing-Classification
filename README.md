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

We propose using a recent dataset (Vrbančič, 2020) containing 88,647 labelled 111-dimensional features to predict whether a website is phishing or legitimate. 96 of the features in the dataset contain information extracted from the domain name and IP, and the other 15 contain information about the website itself. Each of the instances have been verified by multiple sources via the industry standard PhishTank registry. 30,647 of the instances are labeled as phishing and the other 58,000 instances are labeled as legitimate.

## Methods

Unsupervised: We begin by running Principal Component Analysis on the feature set to reduce the dimensionality and to optimize our training.

Supervised: We train three models using the reduced feature set, namely a model based on decision trees, one based on neural networks, and one based on an SVM to predict whether unseen examples are phishing or legitimate.

Our entire pipeline can be implemented using scikit-learn and we can perform data visualizations using Seaborn and Matplotlib.

Approach Ideas: [https://towardsdatascience.com/phishing-domain-detection-with-ml-5be9c99293e5](https://towardsdatascience.com/phishing-domain-detection-with-ml-5be9c99293e5)

## Datasets

Kaggle: [https://www.kaggle.com/ahmednour/website-phishing-data-set](https://www.kaggle.com/ahmednour/website-phishing-data-set)

Science Direct: [https://www.sciencedirect.com/science/article/pii/S2352340920313202](https://www.sciencedirect.com/science/article/pii/S2352340920313202)

## Potential Results/Discussion

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
