# Fake News Detection Project <br>

![image alt](https://images.theconversation.com/files/284418/original/file-20190717-173334-1b9vdud.jpg?ixlib=rb-4.1.0&rect=0%2C350%2C6490%2C3240&q=45&auto=format&w=1356&h=668&fit=crop)

This project demonstrates a simple fake news detection model using a Passive Aggressive Classifier and TF-IDF vectorization.

## Project Steps:

1.  **Load Data**: The project starts by loading the news data from a CSV file into a pandas DataFrame. <br>
```python
df = pd.read_csv('/content/news.csv')
print(df.shape)
display(df.head())

# Get accurate labels from dataframe
labels=df.label
labels.head()
```
Below is the training set used for the model and output:
```
Below is a preview of the dataset used for training and testing the model:

| Unnamed: 0 | Title                                         | Text                                                        | Label |
|------------|-----------------------------------------------|-------------------------------------------------------------|-------|
| 8476       | You Can Smell Hillary’s Fear                  | Daniel Greenfield, a Shillman Journalism Fello...            | FAKE  |
| 10294      | Watch The Exact Moment Paul Ryan Committed... | Google Pinterest Digg Linkedin Reddit Stumbleu...            | FAKE  |
| 3608       | Kerry to go to Paris in gesture of sympathy   | U.S. Secretary of State John F. Kerry said Mon...            | REAL  |
| 10142      | Bernie supporters on Twitter erupt in anger...| — Kaydee King (@KaydeeKing) November 9, 2016 T...            | FAKE  |
| 875        | The Battle of New York: Why This Primary...   | It's primary day in New York and front-runners...            | REAL  |
```
2.  **Split Data**: The data is split into training and testing sets to prepare for model training and evaluation. <br>
```python
# Split into training and testing sets
x_train,x_test,y_train,y_test=train_test_split(df['text'], labels, test_size=0.2, random_state=7)
```
3.  **Vectorize Text**: The text data is transformed into numerical features using TF-IDF (Term Frequency-Inverse Document Frequency) vectorization. This process converts the text into a matrix of token counts, weighted by their frequency. <br>
```python
# Initialize a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english', max_df=0.7)

# Train and transform the training set, and transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)
```
4.  **Train Model**: A Passive Aggressive Classifier is initialized and trained on the vectorized training data. This is a type of online learning algorithm that is suitable for large datasets. <br>
```python
# Initialize the PAC
pac = PassiveAggressiveClassifier(max_iter=50)
pac.fit(tfidf_train, y_train)
```
5.  **Evaluate Model**: The trained model is used to make predictions on the test set, and the accuracy of the model is calculated. A confusion matrix is also generated to understand the performance in terms of true positives, true negatives, false positives, and false negatives. <br>
```python
# Make predictions on the test set and calculate accuracy
y_pred = pac.predict(tfidf_test)
score = accuracy_score(y_test, y_pred)
print(f"Accuracy: {round(score*100,2)}%")
```
Output:
```
Accuracy: 92.58%
```

## Code Highlights:

-   Loading data using `pandas.read_csv()`.
-   Splitting data into training and testing sets using `sklearn.model_selection.train_test_split()`.
-   Initializing and fitting a `TfidfVectorizer` from `sklearn.feature_extraction.text`.
-   Initializing and training a `PassiveAggressiveClassifier` from `sklearn.linear_model`.
-   Evaluating the model using `sklearn.metrics.accuracy_score()` and `sklearn.metrics.confusion_matrix()`.

## Results:

The model achieved an accuracy of **92.2%** on the test set.

|                  | Predicted FAKE | Predicted REAL |
|------------------|----------------|----------------|
| **Actual FAKE**  | 586 (TP)       | 43 (FN)        |
| **Actual REAL**  | 51 (FP)        | 587 (TN)       |
