# Predicting-Customer-Sentiment-for-Clothing-Products-A-Machine-Learning-Approach

![image](https://github.com/user-attachments/assets/912e24a8-2b17-4d2b-b0c7-f5a6faea7071)

Problem Statement:

Customer sentiment plays a crucial role in the retail industry, influencing sales, brand reputation, and customer loyalty. Macy’s, a leading department store, seeks to enhance its understanding of customer opinions by predicting the sentiment of clothing product reviews. With a dataset of 5,000 customer reviews, the goal is to develop and compare multiple classification models to accurately predict sentiment, with a special focus on identifying negative reviews.

By leveraging machine learning techniques, Macy’s aims to proactively detect dissatisfaction, improve product offerings, and enhance customer experiences. Accurately identifying negative sentiment can help the company take timely action—whether through quality improvements, customer service interventions, or strategic marketing adjustments—ensuring continued customer trust and satisfaction.

Building and Evaluating a Sentiment Classification Pipeline

After data preprocessing, next step is to develop a classification pipeline to predict sentiment ratings from customer reviews. The analysis will incorporate multiple machine learning models, including Complement Naive Bayes, Logistic Regression, Support Vector Machines, Decision Tree, AdaBoost, Random Forest, and Multilayer Perceptron (MLP/ANN).
The pipeline will follow these key steps for each model:

Preprocess and tokenize review text by removing stop words and setting a minimum document frequency of 5 (min_df = 5).

Transform text data into a TF-IDF-weighted Document-Term Matrix.

Train the classification model using the transformed data.

Evaluate model performance by obtaining and printing key performance metrics.

This structured approach ensures a robust comparison of different classification techniques, helping to identify the most effective model for sentiment prediction.

Goodness of fit and performance for the most suitable classification model Macy's should use to predict customer sentiment

Given Macy’s focus on actionable insights and understanding the drivers of negative customer sentiment, Logistic Regression is the preferred model.

This is a good model because:

Overall Performance: Accuracy = 0.85 - This indicates that the model correctly predicts positive and negative sentiments in 85% of cases, showing it performs well overall.

ROC AUC = 0.92 - The high ROC AUC score demonstrates the model's strong ability to distinguish between positive and negative reviews.

Focus on Negative Sentiment (0): Precision = 0.88 - This means that 88% of reviews predicted as negative are actually negative. This is critical for minimizing false positives, ensuring Macy's does not mistakenly act on reviews that aren’t negative.

Recall = 0.42 - The model correctly identifies 42% of negative reviews. While lower than ideal, it ensures that some actionable insights are captured for business decisions.

While SVM, a 'black box' model demonstrates slightly better metrics, e.g., higher recall for positive reviews and marginally better ROC AUC and Precision/Recall AUC, Logistic Regression model strikes a good balance between performance and interpretability, ensuring that Macy’s can both predict negative sentiment effectively and derive meaningful insights to improve their offerings.

Handling Class Imbalance with Weighted Classification Models

Class imbalance is a common challenge in sentiment analysis, where certain sentiment categories (e.g., negative reviews) may be underrepresented in the dataset. To address this, we will adjust class weights for applicable classification models (Decision Trees, Random Forest, Logistic Regression, and Support Vector Machines).
Balanced class weights assign a weight to each class level inversely proportional to its frequency using the formula:

len(y_train)/(len(np.unique(y_train)) * np.bincount(y_train)).

In this step, we will modify our classification pipeline by incorporating class_weight='balanced' for the models that support it.

The pipeline will:

-Preprocess and tokenize the review text by removing stop words and setting the minimum document frequency to 5 (min_df = 5)

-Transform the tokenized text into a TF-IDF-weighted Document Term Matrix

-Train the classifcation model using balanced class weights

-Evaluate and print performance metrics to assess the impact of class balancing on model performance.

This approach ensures that the models account for class distribution, improving their ability to detect underrepresented sentiment categories.

Analysis and Findings

![image](https://github.com/user-attachments/assets/9cb2dc0d-c4a7-4909-b9f3-1a00bd7de486)

Macy's should use the Logistic Regression model to predict customer sentiment.

After incorporating balanced class weights, the Logistic Regression model metrics remained relatively consistent with those observed before. This demonstrates the model's robustness and stability in handling class imbalances.

There is a notable improvement in the Recall for the negative class (0), which increased significantly from 0.42 to 0.82, indicating that the model is now much better at identifying negative reviews. This aligns directly with Macy's business goal of predicting negative customer sentiments.
