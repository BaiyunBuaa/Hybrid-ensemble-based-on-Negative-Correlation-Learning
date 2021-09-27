# Hybrid-ensemble-based-on-Negative-Correlation-Learning

This project is for the paper "A hybrid ensemble method with negative correlation learning for regression" by Yun Bai from Beihang University.

In this paper, the author designs a hybrid ensemble method containing 12 common regression models and solve the non-convex optimization problem to automatically select and weight all the 12 sub-models.

To solve the optimization problem, this paper uses the inter-point algorithm through the Gekko optimizer developed by Beal LDR et al.(Processes, 2018). In detail, this paper revises the objective function and adds a penalty term in it. This penalty term is derived from Negative Correlation Learning (NCL).

The experiental results show that our method not only outperforms the single sub-model, but also performs better that ohther benchmarks.

So, try it!

1. Datasets

We use six datasets from Kaggle, you can find them in each of the folder in this project.

Every folder contains the following materials:

.csv is the original dataset from Kaggle;

.py is the python file we use to train on every model and get the predictions;

In folder pkl_data you will find all the predictions by 12 sub-models and the data we preprocessed. The .txt files are parameters obtained by grid search and cross validation.

2. Hybrid ensemble

If you want to get through all the processing, you may use the .py file I mentioned above to try.

If you just want to get to know the hybrid ensemble, you can run the GLgekko_optimize.py and use the predictions in the datasets.

You're welcome to try more datasets with our method to check the prediction accuracy.

Then good luck and feel free to contact me.



