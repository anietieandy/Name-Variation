import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.pipeline import Pipeline
x_train = np.array(['JUSTIN TIMBERLAKE. #nufsaid #grammys','Justin Timberlake. Love how everything is blac and white. Oh shit jay-z', 'Justin Timberlake... Drooling.. As long as I got my suit  tie... #GRAMMYs', 'Justin Timberlake and Jay-Z', 'Justin Timberlake Performance','Justin Timberlake broke my TV its all black and white #grammys', 'Justin Timberlake!!! Aaahhh!! I''ve been waiting for this moment forever!!!!! :) #TheGrammys', 'Black and white for Justin Timberlake....hhhmmm.... #Grammys', 'Justin Timberlake will not be available in technicolor tonight.  #Grammys','The two luckiest guys on one stage. #grammys @jtimberlake and Jay-Z','And Jay-Z! Whaaat! This is awesome. #GRAMMYs','Jay-Z, need I say more #GrammyAwards'])
y_train=[["Justin Timberlake"],["Justin Timberlake", "Jay Z"],["Justin Timberlake"],["Justin Timberlake","Jay Z"],["Justin Timberlake"],["Justin Timberlake"],["Justin Timberlake"],["Justin Timberlake"],["Justin Timberlake"],["Jay Z"],["Jay Z"],["Jay Z"]]
x_test= np.array(['I can''t help it, JT is the bomb! #grammys', 'Be still my heart, JT''s back!!! #Grammys', 'hell yes! #Grammys JT and JAYZ', 'JAY Z AND JUSTIN ARE LIKE ROB A BIGGIE #GRAMMYS @llcoolj','J TIM and JAY Z. hollaaaaa #grammys','YES MY MAN IS BACK! SING JUSTIN! #Grammy','JT and Jigga? That wasnt obvious #Grammys','Jay Z just emits greatness at all times. #Grammys'])
target_names=['Justin Timberlake', 'Jay Z']
mlb = MultiLabelBinarizer()
Y = mlb.fit_transform(y_train)
classifier = Pipeline([('vectorizer', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(x_train, Y)
predicted=classifier.predict(x_test)
all_labels = mlb.inverse_transform(predicted)
for item, labels in zip(x_test, all_labels):
    print('{0} => {1}'.format(item, ', '.join(labels)))
