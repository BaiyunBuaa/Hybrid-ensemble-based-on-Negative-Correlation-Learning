# Hybrid-ensemble-based-on-Negative-Correlation-Learning

This project is for the paper ``A hybrid ensemble method with negative correlation learning for regression" by Yun Bai from Beihang University.

In this paper, the author design a hybrid ensemble method containing 12 common regression models and solve the non-convex optimization problem to automatically select and weight all the 12 sub-models.

To solve the optimization problem, this paper uses the inter-point algorithm through the Gekko optimizer developed by Beal LDR et al.(Processes, 2018). In detail, this paper revises the objective function and adds a penalty term in it. This penalty term is derived from Negative Correlation Learning (NCL).

The experiental results show that our method not only outperforms the single sub-model, but also performs better that ohther benchmarks.

So, try it!

