import pickle
from sklearn.externals import joblib
import json
import numpy as np
import warnings
import os
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


#creating a folder to save all the models(one model for one genre)
if not os.path.exists('models'):
    os.makedirs('models')

def get_output_tags_list(data): #Collecting the list of genres from the dataset
    output_tags = []
    for i in data:
        for j in data[i][1]:
            if j not in output_tags:
                output_tags.append(j)
            else:
                pass
    output_tags.sort()
    return output_tags


def get_y_vector_by_tag(data, tag): #Since we are generating 22 different classifier, we are creating output vectors for each one of them
    y = []
    for i in data:
        if tag in data[i][1]:
            y.append(1)
        else:
            y.append(0)
    y = np.array(y)
    return y



vectorizer= TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')

ob = open("train/movies2.json")
data = json.load(ob)

x_list=[]
count=0

for i in data: #Creating a list of plot summaries to be converted into training vector
    x_list.append(data[i][2])

X_train_counts = vectorizer.fit_transform(x_list) #Bag of words representation
joblib.dump(vectorizer,"models/"+'vectorizer.pkl')

print "Shape of training data", X_train_counts.shape
feature_names = vectorizer.get_feature_names()


target_names = get_output_tags_list(data)
print 'Genre tags considered: ', len(target_names), 'Tags:', target_names

target_dic = {}
for tag in target_names:            # Generating y vector for each classifier
    target_dic[tag] = get_y_vector_by_tag(data, tag)


#####Genrating seprate classifier for each tag#############

for tag in target_dic:
    clf2 = LinearSVC(loss='l2', penalty='l2', dual=False, tol=1e-3)  #Secondary Classifer
    y2 = target_dic[tag]
    clf2.fit(X_train_counts, y2)
    print 'Generating classifer to identify movies with '+tag+' genre'
    joblib.dump(clf2, "models/" + str(tag)+'.pkl')