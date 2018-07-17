#!/usr/bin/python

import os
import pickle
import re
import sys
# from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append( "../tools/" )
from parse_out_email_text import parseOutText

"""
    Starter code to process the emails from Sara and Chris to extract
    the features and get the documents ready for classification.

    The list of all the emails from Sara are in the from_sara list
    likewise for emails from Chris (from_chris)

    The actual documents are in the Enron email dataset, which
    you downloaded/unpacked in Part 0 of the first mini-project. If you have
    not obtained the Enron email corpus, run startup.py in the tools folder.

    The data is stored in lists and packed away in pickle files at the end.
"""


from_sara  = open("from_sara.txt", "r")
from_chris = open("from_chris.txt", "r")

from_data = []
word_data = []

### temp_counter is a way to speed up the development--there are
### thousands of emails from Sara and Chris, so running over all of them
### can take a long time
### temp_counter helps you only look at the first 200 emails in the list so you
### can iterate your modifications quicker
temp_counter = 0


for name, from_person in [("sara", from_sara), ("chris", from_chris)]:
    for path in from_person:
        ### only look at first 200 emails when developing
        ### once everything is working, remove this line to run over full dataset
        # temp_counter += 1
        if temp_counter < 200:
            path = os.path.join('..', path[:-1])
            # print path
            print '.',
            email = open(path, "r")

            ### use parseOutText to extract the text from the opened email
            extracted_text = parseOutText(email)

            ### use str.replace() to remove any instances of the words
            ### ["sara", "shackleton", "chris", "germani"]
            # cleaned_text = re.sub(
            #     r"\s+(sara|shackleton|chris|germani)\s+",
            #     ' ',
            #     ' ' + extracted_text + ' '
            # )
            # cleaned_text = re.sub(
            #     r"\s+",
            #     ' ',
            #     extracted_text
            # )
            # cleaned_text = cleaned_text.strip()
            cleaned_text = extracted_text
            cleaned_text = cleaned_text.replace("sara", "")
            cleaned_text = cleaned_text.replace("shackleton", "")
            cleaned_text = cleaned_text.replace("chris", "")
            cleaned_text = cleaned_text.replace("germani", "")
            cleaned_text = cleaned_text.replace("sshacklensf", "")
            cleaned_text = cleaned_text.replace("cgermannsf", "")

            ### append the text to word_data
            word_data.append(cleaned_text)

            ### append a 0 to from_data if email is from Sara, and 1 if email is from Chris
            if name == "sara":
                from_data.append(0)
            elif name == "chris":
                from_data.append(1)

            email.close()

print "emails processed"
print "word_data[152]:", word_data[152]
from_sara.close()
from_chris.close()

pickle.dump( word_data, open("your_word_data.pkl", "w") )
pickle.dump( from_data, open("your_email_authors.pkl", "w") )

### in Part 4, do TfIdf vectorization here
# stop_words = set(stopwords.words('english'))
# word_tokens = word_tokenize(word_data[152])
# filtered_sentence = [w for w in word_tokens if not w in stop_words]
tfidf = TfidfVectorizer(input='content', stop_words='english')
transformed_data = tfidf.fit_transform(word_data)
feature_names = tfidf.get_feature_names()
print "Number of feature names:", len(feature_names)
print "Number they got:", 38757
print "feature_names[34597]:", feature_names[34597]

print "feature 32134:", feature_names[32134]
