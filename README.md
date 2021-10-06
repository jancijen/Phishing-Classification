# Welcome to ML 7641 (Fall 2021) - Project Group 20

Team Members:
1. Tusheet Sidharth Goli - tgoli3
2. Tejas Rajamadam Pradeep – tpradeep8
3. Tillson Thomas Galloway – tgalloway7
4. Jan Jendrusak – jjendrusak3
5. Jiaxuan Chen – jchen813

# Project Proposal

## Project Name

Phishing Classification

## Introduction/Background
In 2020, phishing was the most prevalent type of cybercrime, costing the American public over $54 million in losses. While employee training against phishing has increased in past years, attack sophistication has increased alongside it. A common cyber kill chain involving phishing begins with an email, text, or phone call that directs the victim to a website to harvest credentials, credit card numbers, and sensitive information. Preemptive identification of a website as malicious would allow system administrators to block or warn clients against imminent cyber threats.

## Problem Definition

We propose using a recent dataset (Vrbančič, 2020) containing 88,647 labelled 111-dimensional features extracting information from domain names, IPs, and webpage content to predict whether a website is phishing or legitimate.

## Methods

We begin by running Principal Component Analysis on the feature set to reduce the dimensionality and to optimize our training. We then train three models using the reduced feature set, namely a model based on decision trees, one based on neural networks, and one based on an SVM to predict whether unseen examples are phishing or legitimate. Both of these can be implemented using scikit-learn.

## Potential Results/Discussion

We hope to obtain an accurate model that can classify if a website is phishing or not. Furthermore, we hope to identify which of our models performs better and to analyze the cases in which each excels and fails.

Additionally, we could contribute to the interpretability of our model by characterizing adversarial examples that result in misclassification. We could then perform novel work by analyzing the effect of a monotonicity property (Romeo, 2018) on the Decision Tree classifier in an attempt to increase the difficulty of an adversarial attack.

Our work is directly useful in a real-world situation: it could be implemented in browsers to warn/block users or email clients to analyze incoming/outgoing mail for spam classification.

## References

FBI (2020). Internet Crime Report 2020. Federal Bureau of Investigation. https://www.ic3.gov/Media/PDF/AnnualReport/2020_IC3Report.pdf

Vrbančič, Grega, et al. “Datasets for Phishing Websites Detection.” Data in Brief, vol. 33, Dec. 2020, p. 106438. ScienceDirect, https://doi.org/10.1016/j.dib.2020.106438.

Íncer Romeo, Íñigo, et al. “Adversarially Robust Malware Detection Using Monotonic Classification.” Proceedings of the Fourth ACM International Workshop on Security and Privacy Analytics, Association for Computing Machinery, 2018, pp. 54–63. ACM Digital Library, https://doi.org/10.1145/3180445.3180449.

## Timeline

TODO
