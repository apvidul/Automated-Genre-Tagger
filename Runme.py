from __future__ import  division
from sklearn.externals import joblib
import Generate_classifiers #Training the 22 classifiers
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)



vectorizer = joblib.load('models/vectorizer.pkl')


#The tags are genrated from the movie data set itself. They ae the most common 22 tags found
tags=[u'action', u'adventure', u'black-and-white', u'comedy', u'coming of age', u'crime', u'drama', u'family', u'fantasy', u'history', u'horror', u'indie', u'music', u'mystery', u'other category', u'romance', u'science fiction', u'sports', u'thriller', u'war film', u'western', u'world cinema']

ch=''

while True:
    if ch=='q' or ch=='Q': #Program execution control
        break
    genres = []  # Final list of genres

    print 'Enter plot:(Aftering copying and pasting the plot from any source, press enter and then type END)'
    inp = ''
    while True:   #Accepting multiline input
        string = raw_input()
        if string == 'END':
            break;
        inp = inp + ' ' + string;

    inp = unicode(inp, errors='ignore')  # Discard items that csnt be intpreted as ASCII
    counts = vectorizer.transform([inp])  # Converts the text into a matrix of word counts which are then scaled based on their frquency in the training data (more frequency elements are scaled down to decrease their impact as they areconsidered less informative)

    for tag in tags: #Selecting classifiers one by one where each classifer employs one vs rest strategy.(for example the 'action' classifier is trained by setting the target of a movie data record containing action tag as 1 and movies without action tags as 0 ). So action classifer will determine if the test data item has the features of an action movie or not. Similarly, a 'crime' classifier will check for elements pertaining to crime genre.
        clf = joblib.load('models/'+str(tag)+'.pkl')
        classifier_output = clf.predict(counts)
        if (classifier_output[0]==1):
            genres.append(tag)

    print "The predicted genres of the movie are: ", genres
    print "press 'q' to quit the program or any other key to continue:"
    ch = raw_input()


