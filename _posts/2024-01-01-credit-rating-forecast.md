---
title: Corporate Credit Rating Forecast using Machine Learning Methods
permalink: /projects/credit-rating/
layout: single
read_time: true
comments: true
share: true
author_profile: true
toc: true
toc_label: "Table of Contents"
categories:
 - quant_finance
tags:
 - XGBoost
 - credit_risk
 - classification
 - smote
 
---

[![](https://img.shields.io/badge/GitHub-View_Repository-blue?logo=GitHub)](https://github.com/monishagopalan/credit-rating-forecast)

# Introduction
Corporate credit ratings, issued by credit rating agencies like Standard and Poor's and Moody's, express the agency's opinion about the ability of a company to meet its debt obligations.  Typically, ratings are expressed as letter grades that range, for example, from ‘AAA’ to ‘D’ to communicate the agency’s opinion of relative level of credit risk.

Credit Ratings are not absolute measures of Default Probability. For example, a corporate bond that is rated ‘AA’ is viewed
by the rating agency as having a higher credit quality than a corporate bond with a ‘BBB’ rating. But the ‘AA’ rating isn’t a
guarantee that it will not default, only that, in the agency’s opinion, it is less likely to default than the ‘BBB’ bond.

Each agency applies its own methodology to measure creditworthiness and this assessment is an expensive and complicated process. Usually, the agencies take time to provide new ratings and update older ones. This causes delays in decision-making process for investors who use these ratings to assess their credit risk. 

One solution to address delays would be to use the historical financial information of a company to build a predictive quantitative model capable of forecasting the credit rating that a company will receive. In this project, I employ machine learning techniques, creating classification models that quickly forecast credit ratings. 

# Dataset

The [Corporate Credit Rating](https://www.kaggle.com/datasets/agewerc/corporate-credit-rating/data) dataset is obtained from Kaggle. The dataset loaded as a pandas dataframe, `ratings_df` consists of 2029 entries (rows) and 31 columns. Each entry represents a big US firm traded on NYSE or Nasdaq. The ratings span the period from 2010 to 2016. The dataset has 593 unique US firms, as seen from `ratings_df.Name.value_counts()`.    

## Credit Ratings
The target variable is the `Rating` column, representing the credit rating assigned by agencies. Taking a closer look at the list of agencies and their different ratings using `ratings_df['Rating Agency Name'].value_counts()` and `ratings_df.groupby('Rating Agency Name')['Rating'].unique()`:

![Rating Distribution by Agency](/assets/images/credit-rating/rating-distribution-by-agency.png)

The dataset shows an imbalance in credit ratings, with varying frequencies for each rating category as it is evident from `ratings_df.Rating.value_counts()`.

In addition, we are working with ratings from different agencies. One way to address this is to simplify and merge the ratings labels according to the following table from [Investopedia: Corporate Credit Ratings](https://www.investopedia.com/terms/c/corporate-credit-rating.asp)

| Moody's     | Standard & Poor's |  Fitch            |   Grade      | Risk         |
|-------------|-------------------|-------------------|--------------|--------------|
| Aaa         | AAA               | AAA               | Investment   | Lowest Risk  |
| Aa          | AA                | AA                | Investment   | Low Risk     |
| A           | A                 | A                 | Investment   | Low Risk     |
| Baa         | BBB               | BBB               | Investment   | Medium Risk  |
| Ba, B       | BB, B             | BB, B             | Junk         | High Risk    |
| Caa/Ca      | CCC/CC/C          | CCC/CC/C          | Junk         | Highest Risk |
| C           | D                 | D                 | Junk         | In Default   |

Instead of 10 different rating categories, we have now 6 categories. Using a dictionary for the mapping of new ratings and old ratings and `ratings_df['Rating'].map(rating_dict)` we have

|(New) Rating    | Counts | (Old) Ratings   |
|----------------|--------|-----------------|
|'Lowest Risk'   |  7     |'AAA'            |
|'Low Risk'      |  487   | 'AA', 'A        |
|'Medium Risk'   |  671   |'BBB'            |
|'High Risk'     |  792   |'BB', 'B'        |
|'Highest Risk'  |  71    |'CCC' 'CC', 'C'  |
|'In Default'    |  1     |'D'              |

The rows with the Ratings: `'Lowest Risk` and `'In Default` are dropped from the dataset given their small value counts. 

The categorical variable `Rating` was converted into numerical labels using the `LabelEncoder` from scikit-learn's preprocessing module, assigning a distinct integer code to each unique label.
 
## Input Features
The other columns in the dataset are the input features related to financial indicators and information about the company. 

The 5 features with the company information such as `Name`, `Symbol` (for trading), `Rating Agency Name`, `Date`, and `Sector` provide context and additional details for analysis but their inclusion in the model may not be necessary for the specific task of credit rating prediction.  However, it is important to include the `Sector` feature in our model is crucial for capturing industry-specific nuances that significantly influence a company's financial performance and risk profile. Different sectors exhibit distinct economic characteristics and respond differently to market conditions. By incorporating the `Sector` variable, we aim to enhance the granularity of our analysis, ensuring that the machine learning model discerns sector-specific trends and challenges.

In order to integrate the categorical data `Sector` it into the model, `LabelEncoder` is employed to represent it numerically.

The dataset includes 25 financial indicators that can be categorized into different groups. These financial indicators collectively provide a comprehensive view of a company's financial health and performance, contributing to the evaluation of its creditworthiness.

**(I) Liquidity Measurement Ratios**
These ratios provide insights into a company's short-term financial health and ability to meet its immediate obligations.
1. `currentRatio`: Indicates the company's ability to cover short-term liabilities with short-term assets.
2. `quickRatio`: Measures the company's ability to cover immediate liabilities without relying on inventory.
3. `cashRatio`: Reflects the proportion of cash and cash equivalents to current liabilities.
4. `daysOfSalesOutstanding`: Measures the average number of days it takes for a company to collect payment after a sale.

**(II) Profitability Indicator Ratios**
These ratios evaluate a company's ability to generate profits relative to its revenue and investments.

5. `netProfitMargin`: Represents the percentage of profit relative to total revenue.
6. `pretaxProfitMargin`: Measures profitability before taxes are considered.
7. `grossProfitMargin`: Indicates the percentage of revenue retained after deducting the cost of goods sold.
8. `operatingProfitMargin`: Reflects the company's profitability from its core operations.
9. `returnOnAssets`: Gauges how efficiently a company utilizes its assets to generate earnings.
10. `returnOnEquity`: Measures the return generated on shareholders' equity.
11. `returnOnCapitalEmployed`: Assesses the efficiency of capital utilization in generating profits.
12. `ebitPerRevenue`: Measures earnings before interest and taxes relative to revenue.

**(III) Debt Ratios**
These ratios assess the company's leverage and debt management.

13. `debtEquityRatio`: Measures the proportion of debt relative to equity.
14. `debtRatio` : Represents the percentage of a company's assets financed by debt.

**(IV) Operating Performance Ratios**
These ratios focus on operational efficiency and effectiveness.

15. `assetTurnover`: Evaluates how efficiently a company utilizes its assets to generate sales revenue.
16. `fixedAssetTurnover` : Measures the efficiency of generating sales from fixed assets.
17. `payablesTurnover`: Measures the efficiency of a company's payment of its liabilities.

**(V) Cash Flow Indicator Ratios**
These ratios delve into a company's cash flow dynamics, providing insights into its financial sustainability. 

18. `operatingCashFlowPerShare`: Reflects the cash generated by core business operations per share.
19. `freeCashFlowPerShare`: Measures the amount of cash available to shareholders after covering operational expenses and capital expenditures.
20. `cashPerShare`: Represents the amount of cash available per outstanding share.
21. `operatingCashFlowSalesRatio`: Evaluates the percentage of sales revenue converted into cash from operating activities.
22. `freeCashFlowOperatingCashFlowRatio`: Measures the efficiency of converting operating cash flow into free cash flow.
23. `effectiveTaxRate`: Reflects the company's tax efficiency.
24. `companyEquityMultiplier`: Indicates the multiplier effect on equity due to debt
25. `enterpriseValueMultiple`: Evaluates a company's overall value relative to its earnings.


## Descriptive Statistics

![png](/assets/images/credit-rating/corr-matrix.png)

The function `data.describe()` gives statistical descriptions like `mean`, `min`, `max`, `percentiles` of the numerical financial indicators. Comparison of the mean to the median and examining the range between percentiles, there seems to be an indication of the presence of outliers. 

Features exhibit different scales, as evident from the magnitude of mean and standard deviation values. To ensure equal contribution from all features for machine learning algorithms, feature scaling is performed. The Min-Max scaling technique is applied to normalize the numerical values representing financial indicators. 

$x_{scaled} = \frac{x - x_{min}}{x_{max} - x_{min}}$

For each column, the `MinMaxScaler` function from `sklearn.preprocessing` is used to transform the values into a standardized range between 0 and 1. The values are then multiplied by 1000, to amplify the scaled values. 

Additionally, a logarithmic transformation is applied to each value using the `np.log10` function, with a small constant (0.01) added to avoid issues with zero values. This dual transformation approach aims to normalize and potentially enhance the interpretability of the financial indicators in the dataset.

This work in exploring and preparing our dataset, sets the stage for the next phase: deploying different machine learning models to forecast credit ratings. Our problem of corporate credit rating forecast is a supervised multi-class classification task. 


# ML Models
We split our input data into training (80%) and test data (20%) using `train_test_split()` from `sklearn.model_selection`. Then we create separate dataframes for the input features (X) and target lables (y). 

Various machine learning models are employed to forecast corporate credit ratings. 

**1. Logistic Regression:**
Logistic Regression is a linear model used for binary or multiclass classification. It estimates the probability that a given instance belongs to a particular class. It employs the logistic function (sigmoid) to transform a linear combination of input features into a value between 0 and 1. This output represents the probability of belonging to the positive class. A threshold is applied to make the final classification decision.

**2. K-Nearest Neighbors (KNN):** 
KNN is a non-parametric algorithm used for classification. It classifies a data point based on the majority class among its k-nearest neighbors.
The algorithm calculates the distance between the input instance and all data points in the training set. It then assigns the class that is most common among the k-nearest neighbors.

**3. Support Vector Machine (SVM):**
SVM is a powerful classification algorithm that aims to find a hyperplane in a high-dimensional space that best separates data points of different classes. SVM transforms input features into a higher-dimensional space and seeks a hyperplane that maximizes the margin between classes. It classifies instances based on which side of the hyperplane they fall on.

**4. Random Forest:**
Random Forest is an ensemble learning method that constructs a multitude of decision trees during training and outputs the mode of the classes for classification tasks. It builds multiple decision trees using a subset of features and a random subset of the training data. The final prediction is determined by aggregating the predictions of individual trees (voting or averaging).

**5. Gradient Boost:**
Gradient Boosting is an ensemble learning method that builds a series of weak learners (typically decision trees) sequentially, each correcting the errors of its predecessor. It fits a weak model to the residuals of the previous model. This process continues, with each new model focusing on the mistakes of the ensemble. The final prediction is a weighted sum of the predictions from all weak models.

**6. XGBoost:**
XGBoost (Extreme Gradient Boosting) is an advanced version of gradient boosting that incorporates regularization and parallel processing, making it highly efficient. Similar to gradient boosting, XGBoost builds a series of trees sequentially. It optimizes a regularized objective function, combining the prediction from each tree. It uses a technique called boosting to strengthen the model iteratively.

The models are then trained on the training dataset `(X_train, y_train)` and evaluated on the test dataset `(X_test, y_test)`. The accuracy of each model is calculated using the `metrics.accuracy_score` function from `scikit-learn`. 

## SMOTE
The Synthetic Minority Over-sampling Technique (SMOTE) is employed to address the issue of class imbalance in the training dataset. The training dataset is characterized by a disproportionate representation of certain credit rating classes as seen from `y_train.value_counts()`, and this can lead to biased model predictions. The SMOTE algorithm is applied to generate synthetic instances of the minority class, thereby balancing the distribution of ratings. By doing so, it ensures that the machine learning models are exposed to a more representative training set, enhancing their ability to generalize and make accurate predictions across different credit rating categories. The resulting distribution of ratings in the resampled training set demonstrates the effectiveness of SMOTE in mitigating class imbalance. Now, we train the models again with the resampled dataset `(X_train_resampled, y_train_resampled)` and evaluate it on `(X_test, y_test)`. 

![Accuracy by Model with and without SMOTE](/assets/images/credit-rating/smote-accuracy.png)

It is observed that the accuracy slightly decreased after employing SMOTE. This phenomenon could be attributed to the introduction of synthetic samples, potentially leading to increased noise and complexity in the dataset. Moreover, the choice of sampling strategy in SMOTE, such as `'auto'`, might have not been optimal for our problem. Alternative methods, including undersampling or experimenting with different SMOTE sampling strategies, could be considered. Keeping in mind that the decision to use or omit SMOTE is usually guided by a comprehensive analysis of the specific dataset characteristics and the performance trade-offs associated with different sampling techniques, we can omit SMOTE resampling in our case.

## Hyperparameter Optimisation
We perform hyperparameter optimization for the XGBoost model. 

`RandomizedSearchCV` is a method for hyperparameter tuning that explores a defined number of random combinations of hyperparameters. In our case `n_iter=15` and it searches through 15 different combinations. `StratifiedKFold` is employed for cross-validation. It ensures that each fold preserves the same distribution of target classes as the entire dataset, which is crucial for maintaining the representation of different credit ratings in each fold.  

`xgb_params` defines a grid of hyperparameters to search through. XGBoost parameters such as learning rate, number of estimators, maximum depth, minimum child weight, subsample, and colsample by tree are considered. The metric used for evaluation is accuracy `(scoring='accuracy')`. The search is parallelized `(n_jobs=-1)`, making use of all available CPU cores. The model is fitted to the training data to identify the best hyperparameters. The best model with optimal hyperparameters is then extracted from the search results. Finally, the best model is evaluated on the test set, and the accuracy is noted.


# Results

![Accuracy by Model](/assets/images/credit-rating/accuracy-model.png)

The `classification_report` provides a comprehensive summary of key classification metrics, including accuracy, precision, recall, and F1 score. 

- **Accuracy** reflects the overall correctness of the model's predictions. 
- **Precision** measures the proportion of true positive predictions among instances predicted as positive, highlighting the model's ability to avoid false positives. 
- **Recall** gauges the model's effectiveness in capturing true positives among all actual positive instances. 
- **F1 score** balances precision and recall, providing a single metric that considers both false positives and false negatives. 


`confusion_matrix` provides insights into the performance of a classification model. It helps to understand the distribution of true positive, true negative, false positive, and false negative predictions. This information aids in assessing the model's ability to correctly classify instances into different credit rating categories.

![Confusion Matrix](/assets/images/credit-rating/confusion-matrix.png)

The feature importance plot for XGBoost illustrates the contribution of each input feature to the model's decision-making process. This is particularly useful in understanding which financial indicators or ratios play a significant role in predicting credit ratings. Analyzing feature importance guides financial analysts and stakeholders in focusing on key metrics that heavily influence the model's predictions.

![Feature Importance](/assets/images/credit-rating/feature-importance.png)


# References

1. Parisa Golbayani, Ionuţ Florescu, Rupak Chatterjee: A comparative study of forecasting corporate credit ratings using neural networks, support vector machines, and decision trees. _The North American Journal of Economics and Finance,
Volume 54, 2020_, https://doi.org/10.1016/j.najef.2020.101251.

2. https://www.kaggle.com/datasets/agewerc/corporate-credit-rating/

3. [Guide to Credit Rating Essentials](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=&cad=rja&uact=8&ved=2ahUKEwje_Nn6i72DAxWN8gIHHbgjDfAQFnoECBcQAQ&url=https%3A%2F%2Fwww.spglobal.com%2Fratings%2F_division-assets%2Fpdfs%2Fguide_to_credit_rating_essentials_digital.pdf&usg=AOvVaw1eCCN2ZJXYlPuBycn7-Rmi&opi=89978449)