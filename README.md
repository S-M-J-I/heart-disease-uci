# Heart Disease UCI
A short task of predicting heart disease and finding out the perfect model for it using the [UCI Heart Disease dataset](https://www.kaggle.com/datasets/redwankarimsony/heart-disease-data)
<br>

[Kaggle version](https://www.kaggle.com/code/smjishanulislam/task-1/notebook)

### Technologies used:
- Jupyter Notebook
- Numpy
- Pandas
- Scikit Learn

### Dataset:
- `id` (Unique id for each patient)
- `age` (Age of the patient in years)
- `origin` (place of study)
- `sex` (Male/Female)
- `cp` chest pain type ([typical angina, atypical angina, non-anginal, asymptomatic])
- `trestbps` resting blood pressure (resting blood pressure (in mm Hg on admission to the hospital))
- `chol` (serum cholesterol in mg/dl)
- `fbs` (if fasting blood sugar > 120 mg/dl)
- `restecg` (resting electrocardiographic results) => Values: [normal, stt abnormality, lv hypertrophy]
- `thalach` maximum heart rate achieved
- `exang` exercise-induced angina (True/ False)
- `oldpeak` ST depression induced by exercise relative to rest
- `slope` the slope of the peak exercise ST segment
- `ca` number of major vessels (0-3) colored by fluoroscopy
- `thal` [normal; fixed defect; reversible defect]
- `num` the predicted attribute

The target column is the `num`. Since `num` can only occur within 5 categories (as shown by the dataset), it is a multi-class classification problem.

The features are all columns EXCEPT `id` and `origin`.

### Process:
At first we check the amount of missing values we have in our dataset. We then drop the columns that will have no significant affect in our predictions such as `id` and `origin`. If there are any missing labels, we drop that entire row.

We then split the data into features(x) and labels(y).

We then fill the missing features. The numerical data are filled with the mean of all the data in their respective columns. The categorical data are filled with the modal value that occur in their respective columns.

After we've handled missing data, we impute (turn non-numerical data into numerical) our dataset. Here, I've used the `sklearn.preprocessing.OneHotEncoder` for the imputation.

After our data has been imputed, we spilt the data into training and test sets. I've considered the test size to be 20% of the total data.

We then use 6 different models on the data to find which one gives us the best case:
- `Linear Model` (Accuracy score: 0.625)
- `Support Vector Machine` (Accuracy score: 0.5326)
- `Nearest Neighbours` (Accuracy score: 0.5)
- `Naive Bayes` (Accuracy score: 0.5217)
- `Decision Trees` (Accuracy score: 0.4565) **Worst performing**
- `Random Forest` (Accuracy score: 0.6630) **Best performing**

### Conclusion:
The best model for this case, a multi-class classification problem, is the **Random Forest** (RandomForestClassifier), having an accuracy score of 66%. 

Random Forest is suitable for situations when we have a large dataset, and interpretability is not a major concern. It also provides very high accuracy. 

However, the main limitation of random forest is that a large number of trees can make the algorithm too slow and ineffective for real-time predictions. In general, these algorithms are fast to train, but quite slow to create predictions once they are trained.
