# Sleep-Apnea-Prediction

1.	Abstract

Millions of people worldwide suffer from sleep apnea, which frequently results in serious health issues and a lower quality of life. To lessen its negative effects on wellbeing, early detection and intervention are essential. In this work, we introduce a thorough sleep apnea prediction system that analyzes physiological data using cutting-edge machine learning and deep learning methods. To create a reliable dataset, we combine information gathered from clinical-grade sensors, such as those for electrocardiogram activity, blood oxygen levels, heart rate, and background noise. This experiment shows how well deep learning techniques and conventional machine learning algorithms can be combined to produce accurate sleep apnea predictions, underscoring the promise of data-driven approaches to enhance diagnostic capabilities and progress the area of sleep health.

Keywords: Sleep Apnea, Machine Learning, Deep Learning, Physiological Data, Electrocardiogram (ECG), Blood Oxygen Levels, Heart Rate Monitoring, Background Noise Analysis, Data-Driven Diagnostics, Sleep Health Prediction

2.	Introduction

Sleep is an essential biological process that influences overall health and well-being. Despite its significance, millions of people worldwide suffer from sleep problems including sleep apnea, which are becoming more widespread. Particularly, sleep apnea is a disorder marked by frequent breathing pauses while you sleep. 
This illness can cause major health issues like high blood pressure, heart disease, and cognitive decline. Sleep apnea has a major negative influence on quality of life, productivity, and mental health if it is not identified and treated.

Our project's goal is to create a trustworthy system for predicting sleep apnea by utilizing a combination of machine learning and deep learning approaches. We investigate various computational approaches to increase diagnostic accuracy by using algorithms including Decision Tree, Random Forest, Support Vector Machines (SVM), XGBoost, and Artificial Neural Networks (ANN), in addition to ensemble techniques and K-Means clustering.
This method offers a novel, scalable, and effective way to diagnose sleep apnea, and it has the potential to revolutionize the way that sleep disorders are identified and treated.

3.	Existing Methods/Algorithms

[1]To overcome the difficulty of detecting sleep apnea, a number of devices have been created, including ones that enable examination and analysis to be done in the patient's home. In order to detect patterns linked to sleep apnea, these systems frequently use sensors to record physiological data, which are subsequently automatically examined by algorithms.

[2]In order to improve classification performance by utilizing the complimentary capabilities of separate classifiers, some studies have suggested merging multiple classifiers. For example, classifier combinations like AdaBoost with Decision Stump, Bagging with REPTree, and kNN or Decision Table have been investigated to increase detection accuracy using specific SpO₂ and ECG variables.

[3]For the purpose of detecting sleep apnea, numerous studies have used individual biological markers, such as SpO₂, ECG, EOG, or EEG. SpO₂ and ECG signals are commonly highlighted among them because of their substantial association with apneic episodes, which makes them trustworthy indicators in sleep research.

[4]The development of automated classification systems that handle brief ECG data epochs has also been the subject of research. Support Vector Machines (SVM), which have been trained and evaluated on sleep apnea recordings, are one such method that shows promise in accurately identifying apneic episodes.

4.About Dataset

In order to predict sleep apnea, the dataset employed in this experiment includes a variety of lifestyle and health characteristics. Each record contains personal information like gender, age, and occupation and represents an individual who is uniquely recognized by a Person_ID. Sleep_Duration (the average number of hours of sleep per night) and Quality_of_Sleep (a subjective evaluation indicating how restful sleep is) are used to record sleep-related data. Additional health metrics that may affect the quality of sleep include Physical Activity Level, Stress Level, BMI Category, Blood Pressure, Heart Rate, and Daily Steps. 

Because snoring is frequently linked to sleep apnea, the dataset additionally includes SPO2_Rate, which measures blood oxygen levels (a crucial sign in determining sleep apnea), and Snoring, a binary indicator of whether the individual snores. The target variable, Sleep_Disorder, is the last column and indicates whether a sleep problem has been identified or not. When combined, these characteristics offer a thorough basis for creating a reliable and accurate sleep apnea prediction model through the use of cutting-edge machine learning and deep learning methodologies.

Table 1: The head values of the data set
![image](https://github.com/user-attachments/assets/3adbe2c6-0697-4652-88b5-441aa6a5ccf4)

5.Proposed Methodology/Algorithm

The suggested method for predicting sleep apnea is a thorough process that analyzes a dataset made up of many physiological, behavioral, and lifestyle factors using a variety of machine learning and deep learning approaches. To guarantee reliable and accurate predictions, the procedure includes data preparation, feature engineering, model training, evaluation, and ensemble approaches.

1.	Data preprocessing: 
Using the proper imputation techniques to handle missing or inconsistent values.
For consistent scaling, continuous features like blood pressure, heart rate, and SpO₂ values should be normalized or standardized.
 Gender, employment, and BMI category are examples of categorical variables that are encoded into numerical representations that machine learning algorithms can use.

2.  Feature Selection: 
To enhance model performance, statistical and machine learning-based techniques such as Recursive Feature Elimination (RFE) and correlation analysis are used to identify and choose the most important features (such as SpO₂, heart rate, and snoring).

3.  Model Execution:

Training and assessing a variety of machine learning models, like as
-	Decision Tree: For predictions that are tree-based and interpretable.
-	Random Forest: To use ensemble learning to enhance generalization.
-	Support Vector Machines: For classifying high-dimensional data.
-	XGBoost: To achieve high accuracy and handle unbalanced data.
-	K-Means Clustering: for finding patterns in the dataset without supervision.
-	Artificial Neural Networks: for predictions based on deep learning.

4.  Ensemble Learning: 
To improve accuracy and robustness, combine the predictions of several models using ensemble approaches like stacking and voting.
By using weighted averaging, models that perform better are given more weight.

5. Evaluation measures:
To guarantee accurate predictions, the models' performance is assessed using measures including accuracy, precision, recall and F1-score.

6. Prediction and Insights:
Examining the ensemble model's output to forecast sleep apnea.
Giving practical information about important contributing factors (such as snoring, sleep duration, and SpO₂ levels) for possible medical interventions.
 
6. Results and Discussions

1.	Random Forest:
   
![image](https://github.com/user-attachments/assets/7c697232-caa5-4e01-bae2-7230329e43c7)

![image](https://github.com/user-attachments/assets/df3a54f3-ed4d-4075-9451-1638db745609)

![image](https://github.com/user-attachments/assets/2c6cc02f-03b2-480f-b22e-e0a3b3d9436c)

With an accuracy of 84% across all classes, the Random Forest classifier showed good overall performance in predicting sleep disorders. With perfect recall (1.00) and good precision (0.92), the model demonstrated especially remarkable performance in identifying cases of sleep apnea, yielding an outstanding F1-score of 0.96. This implies that the model is very accurate at identifying those who have sleep apnea. The model maintained strong performance with balanced precision (0.86) and recall (0.89) for typical sleep patterns, resulting in an F1-score of 0.88. With an F1-score of 0.53 and lesser precision (0.62) and recall (0.47), the model was comparatively less successful at identifying insomnia. In line with clinical knowledge of the indicators, feature importance analysis showed that Snoring was the most significant predictor, followed by SPO2_Rate and Heart_Rate.

2.	Decision Tree:
   
![image](https://github.com/user-attachments/assets/79ab8058-3bae-47bb-9643-2c1c1fce55de)

![image](https://github.com/user-attachments/assets/973f370d-cf6e-4fab-8652-04bb77ecd348)

![image](https://github.com/user-attachments/assets/024ba162-4f91-4f37-a3e8-a8cf34a75f93)

The Decision Tree classifier's efficacy as a diagnostic tool was demonstrated by its strong overall accuracy of 81% in predicting sleep disorders. SPO2_Rate was the most significant predictor (importance = 0.52), followed by Heart_Rate (importance ≈ 0.38), and Snoring (importance ≈ 0.10), according to the model's feature importance analysis. Blood oxygen saturation values are important markers of sleep apnea, which is consistent with medical knowledge on hierarchical priority. The model performed exceptionally well in identifying this crucial condition, as evidenced by the confusion matrix, which indicates that it correctly detected all 22 occurrences of sleep apnea (as shown by the bottom-right cell). With a weighted average F1-score of 0.81, the model demonstrated balanced performance across classes, while it could still do better at differentiating between insomnia and regular sleep patterns.

3.	K-Means Clustering:
   
![image](https://github.com/user-attachments/assets/464b37ca-4512-438b-a773-08368c9fe74b)

![image](https://github.com/user-attachments/assets/e428bae6-35fe-4bb0-a187-1fbe4e769ba8)

Different patterns in the classification of sleep disorders were found using the K-means clustering technique, which was especially noticeable in the correlation between important physiological indicators. Three unique clusters can be seen in the scatter plot matrix, with significant differences in the SPO2_Rate, Heart_Rate, and Snoring patterns. There is a distinct cluster separation in the SPO2_Rate distribution, with one cluster concentrated around higher values (95–100%), which probably represent typical sleep patterns, and another cluster with lower values (75–85%), which would indicate cases of sleep apnea. A bimodal pattern can be seen in heart rate distributions, with clusters concentrated around 70–75 bpm and 80–85 bpm. In order to distinguish between sleep apnea and typical cases, the Snoring feature shows clear grouping, with one cluster displaying high snoring scores (about 1.0) and another with lower values (around 0).

4.	SVM:
   
![image](https://github.com/user-attachments/assets/3de067be-2e3e-464e-aee5-4e01505a2bd8)

With a high classification accuracy of 92% for sleep disorders, the SVM classifier showed outstanding overall performance. With exceptional precision (0.95) and recall (0.98), the model demonstrated exceptional performance in class 1 (identifying normal sleep patterns), yielding the highest F1-score of 0.97 among all classes. The model produced an F1-score of 0.85 for the insomnia cases (class 0) by striking a reasonable balance between precision (0.82) and recall (0.88). The identification of sleep apnea (class 2) has an F1-score of 0.87 due to its high precision (0.93) and somewhat lower recall (0.81). Just two cases of sleep apnea were mistakenly categorized as insomnia, and only one normal case was misclassified as both insomnia and sleep apnea, according to the confusion matrix.

5.	ANN:
   
![image](https://github.com/user-attachments/assets/61720469-f54b-4d70-911c-9caeb446d58b)

The training curves showed that the Artificial Neural Network exhibited robust learning progression and convergence in the classification of sleep disorders. The model's training accuracy (purple line) exhibited continuous growth, starting from roughly 50% and slowly growing to approximately 90-92% by the end of training. After the first epochs, the validation accuracy (green line) stabilized at about 88%, showing good generalization without overfitting. It closely followed the training accuracy. Effective model optimization was suggested by the training loss's (orange line) ideal exponential decay pattern, which began at roughly 1.0 and rapidly decreased over the first 10 epochs before stabilizing at 0.2–0.3.

6.	XGBOOST:
   
![image](https://github.com/user-attachments/assets/7e800a6a-fa79-4903-ab5c-3db974c6ed4a)

With the greatest overall accuracy of 93% among all investigated models, the XGBoost classifier showed exceptional performance in the categorization of sleep disorders. With a precision of 0.95 and recall of 0.98 (F1-score: 0.97), the model demonstrated exceptionally balanced performance across all classes, with class 1 showing particularly good results in recognizing regular sleep patterns. While insomnia categorization (class 0) maintained consistent metrics with both accuracy and recall at 0.88 (F1-score: 0.88), sleep apnea detection (class 2) demonstrated solid performance with 0.93 precision and 0.88 recall (F1-score: 0.90). With just three misclassifications overall across all classes, the confusion matrix shows very little misclassification. Person_ID, SPO2_Rate, and Sleep_Duration emerged as the top predictors, followed by Heart_Rate and Systolic_BP, according to feature importance analysis, which yielded insightful results.

Ensembling:

![image](https://github.com/user-attachments/assets/079d7355-ebcd-49ef-9f50-ba9e74b41ff8)

![image](https://github.com/user-attachments/assets/e92bb3aa-22b5-4fff-8cd0-051882ecfddf)

Clear performance differences amongst the models were shown by the ensemble comparison, with XGBoost emerging as the best classifier with an accuracy of 94.67% and well-balanced metrics (F1-scores: 0.88-0.98). With respective accuracies of 93.33% and 90.67%, Decision Tree and Random Forest likewise shown strong performance. ANN and SVM, on the other hand, performed noticeably worse (63% and 64% accuracy). Due to their capacity to identify intricate links in sleep-related data, the results show that tree-based models—in particular, XGBoost—are the most successful at classifying sleep disorders. Given the significant performance difference between tree-based models and alternative methods, XGBoost ought to be the recommended option for this particular sleep disorder prediction challenge.

7.Conclusion

In this project, we used a dataset with a variety of physiological and behavioral factors to investigate the efficacy of many machine learning and deep learning algorithms for the prediction of sleep apnea. Tree-based algorithms outperformed the other models used; XGBoost had the best accuracy of 94.67%, followed by Random Forest (90.67%) and Decision Tree (93.33%). K-means clustering successfully found natural patterns in the dataset that corresponded to clinical sleep problem categories, but Support Vector Machines (SVM) and Artificial Neural Networks (ANN) had mediocre performance. Important characteristics including SPO2_Rate, Heart_Rate, and Snoring were consistently found to be important predictors in every model, demonstrating their importance in detecting sleep apnea.
Additionally, the research supported medical findings, especially the significance of cardiovascular and oxygen saturation signals in the identification of sleep disorders.

This work highlights how data-driven methods might improve sleep apnea diagnostic skills. Machine learning models, in particular XGBoost, have shown significant promise as trustworthy instruments for early diagnosis and treatment planning. The robustness of the suggested methodology is demonstrated by the balanced performance across various sleep disorder categories. However, to demonstrate the usefulness of these models in actual medical situations, additional validation using bigger and more varied datasets as well as clinical trials are advised. In terms of using cutting-edge computational methods to enhance sleep health outcomes, this work is a major advancement.


