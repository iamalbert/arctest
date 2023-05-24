# Observations

We have job posting data of 17828 rows, 862 (<5%) of which are labelled as fraudulent. This is a typical problem for imbalanced data or partial labelled data. 


## Feature engineering

I like to run a simple test to see the effectiveness of each feature. I chose 8 categorical features to represent each row with one-hot encoding. That is, each row is a 204-dimentional binary vector.

The 8 fields are

- `telecommuting`
- `has_company_logo`
- `has_questions`
- `required_experience`
- `required_education`
- `employment_type`
- `industry`
- `function`


Note:

1. No language processing or text normalization was applied. So there might be duplicates.

2. I did not chose `department` column as it was too sparse. The column has ~1300 differnt values and did not contribute too much.

## Tranditional Models

I chose LogisticRegression and LightGBM as classifiers with the consideration of quick computation, the natural of imbalanced dataset, and the explainitory of the model. The dataset was split 50-50 to training set and testing set. The result is showned in jupyter notebook. It's worth noting that


1. Both model selected the following feature/value to be the top 4 important indicator of a fraud job post.

- `has_questions`: `0`
- `required_education`: `High School or equivalent`
- `has_company_logo`: `0`
- `industry`: `Oil & Energy`


2. I tried with different hyper-parameters such as number of folds of cross-validation, penalty, ratio of training/testing set. They all produced similar results and tendency. I believe it suggested there existed some salient pattern of fradulent job posts.

3. LightGBM produced nice & balanced results, with f1=0.6326 and precision=0.8544. If sensitivy is not an issue, I think I'd deploy this model after some more dedicate tuning.

4. Liblinear sometimes could not converge due to the sparsity of positive labels.


## Deep learning method

I used the same feature set and built a simple deep neural net which had 3 linear layers. The model was easy to train with cross entropy loss and Adam. It coverged very fast. In the most simple settings it already outperformed LightGBM methods. I tried two methods to deal with the inbalance dataset

1. **Weighed loss**. Setting positive class to a bigger wight (10, 20, 30) did not benefit the model as expected. It did force the model to have a high recall but also resulted in low precision and F1.

2. **Label smoothing**. It softened the probabilities of labels by adding a small noise. It was very useful to prevent overfitting on such small dataset. I tried 1%, 5%, and 10% and they all seemed to work in a similar way and slightly better than previous settings. One unexpected suprise was that, models with label smoothing converged rapidly within mere one or two epoches. 

# Results

```
Result(acc=0.9553, f1=0.2866, precision=0.7273, recall=0.1784) LogisticRegressionCV(cv=3, random_state=123, solver='newton-cholesky')
Result(acc=0.9553, f1=0.2866, precision=0.7273, recall=0.1784) LogisticRegressionCV(cv=3, random_state=123, solver='liblinear')
Result(acc=0.8048, f1=0.2927, precision=0.1790, recall=0.8030) LogisticRegressionCV(class_weight='balanced', cv=3, random_state=123, solver='liblinear')
Result(acc=0.6250, f1=0.1970, precision=0.1104, recall=0.9145) LogisticRegressionCV(class_weight={0: 1, 1: 100}, cv=3, random_state=123, solver='liblinear')
Result(acc=0.9729, f1=0.6742, precision=0.8523, recall=0.5576) LGBMClassifier(random_state=123)

Result(acc=0.9748, f1=0.6980, precision=0.8764, recall=0.5799) Deep Learning: CrossEntropyLoss epoch 7
Result(acc=0.9265, f1=0.5293, precision=0.3905, recall=0.8216) Deep Learning: CrossEntropyLoss(weighed) epoch 7
Result(acc=0.9740, f1=0.7036, precision=0.8250, recall=0.6134) Deep Learning: CrossEntropyLoss(label_smoothing) Epoch 1
```


# Deployment

## LightGBM

Deploying such model is rather easy. I'd use Flask to write an endpoint and use Gunicorn to support parallel requests. Note that

1. If you import lightgbm and the model in Flask, gunicorn will duplicate the model and use a lot of memory. I suggest to write a gunicorn-wrapper that loads the model *before* flask. Doing so, the threads and processes spanwed by gunicorn could use the shared memory.

2. Unfortunatly lightgbm do not support async style yet. Async-based framework like fastapi or tornado will experience weird bugs here and there. 

## Pytorch 

Same as LightGBM. Write a Flask app and serve as HTTP API. Alternatively, we can use TorchServe to have a quicker build.

Another way, thanks to my model being rather straightforward, would be exporting the model to be an ONNX model file and then use other langauge (Python, Java, C++, even Javascript in frontend with onnx.js) to expose an API.