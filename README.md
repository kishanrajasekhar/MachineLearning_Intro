# MachineLearning_Intro

This is a project I did with my partner in an introductory machine learning class. Our professor created a private Kaggle competition
for the class. Our task was to predict whether there will be rainfall or not at a certain location. It was a classification problem, 
so we used different classification models. We used KNN, Random Forest, and Adaboost. These models were used from the Python sklearn library. 
Afterwards, the individual models were combined into an ensemble. The performance was measured using the area under the roc curve (ROC-AUC).

These were our results:

Model            | Training ROC-AUC | Validation ROC-AUC | Kaggle Score (Private Leaderboard)

1)KNN	           | 0.867312308946   |	0.706247494793	   | 0.70773

2)Random Forests | 0.968757280831	  | 0.719818134936	   | 0.72541

3)Adaboost	     | 0.724944697866	  | 0.699405989396	   | 0.65361

4)Ensemble       | 0.959052296333	  | 0.741687620014	   | 0.74339


More details can be found in the report (KaggleReport.docx).

Partner: https://github.com/jasian
