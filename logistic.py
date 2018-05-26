import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import GridSearchCV
import nltk
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

def class_metrics(y_test, y_preds):
    print('Accuracy score: ', format(accuracy_score(y_test,y_preds)))
    print('Precision score: ', format(precision_score(y_test, y_preds)))
    print('Recall score: ', format(recall_score(y_test, y_preds)))
    print('F1 score: ', format(f1_score(y_test,y_preds)))
    print("Confusion Matrix: ") 
    print(confusion_matrix(y_test, y_preds))

def run_vanilla_pipe():
    print("Vectorizier/Logistic Regression pipe with default settings: ")
    tfidf = TfidfVectorizer(strip_accents='unicode',
                            lowercase=False,
                            preprocessor=None) 
    lr_vanilla =  Pipeline([('vect', tfidf),
                    ('clf', LogisticRegression())
                    ])
    lr_vanilla.fit(X_train, y_train)
    predictions = lr_vanilla.predict(X_test)
    class_metrics(y_test, predictions)
    

    tfidf = TfidfVectorizer(strip_accents='unicode', ngram_range=(1,2)
                            lowercase=False,
                            preprocessor=None)
    lr_vanilla =  Pipeline([('vect', tfidf),
                    ('clf', LogisticRegression())
                    ])                             
    
#process and split data    
columns = ['comment_text', 'toxic']
data = pd.read_csv('train.csv', usecols = columns)
X = data['comment_text']
y = data['toxic']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42)
    
def grid_search(param_grid): 
    print("Running Grid search on Logistic Regression")     
    tfidf = TfidfVectorizer(strip_accents='unicode',
                            lowercase=False,
                            preprocessor=None) 
    lr_tfidf = Pipeline([('vect', tfidf),
                         ('clf', LogisticRegression(random_state=0))])
    gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                               scoring='f1',
                               cv=5,
                               verbose=2,
                               n_jobs=-1)
    gs_lr_tfidf.fit(X_train, y_train)
    print('Best Parameter Set %s ' % gs_lr_tfidf.best_params_)
    print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)
    clf = gs_lr_tfidf.best_estimator_
    print('Test Accuracy: %.3f' % clf.score(X_test, y_test))      

param_grid = [{'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 4.0, 10.0]}]
               
if __name__ == '__main__':
    run_vanilla_pipe() 
    grid_search(param_grid)               
                          
                               

                      
    