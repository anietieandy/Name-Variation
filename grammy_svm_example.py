import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multiclass import OneVsRestClassifier
x_train = np.array(['taylor sounds so much better when you cant hear her','looks like taylor swift brought her circus to the grammys', 'taylor swift rocking out to mumford & sons, thats whats up', 'taylor was terrific to start the grammys', 'nobody break taylor swifts heart tonight we need her camera ready','taylor sift opening the show', 'feeling especially bad for @harry_styles after seeing taylors performance. if she pinned you 2 a wheel too, tell an adult you trust', 'why did taylor swift open the grammys', 'why is taylor swift opening the grammys when she cant sing live'
                    'Taylor Swift was terrific to start the #GRAMMYS','Taylor Swift, stop pretending to like other peoples music because we all know you only like your own you egocentric bitch. Grammys',
                    'Taylor Swifts robot arm dancing is going to take the nation by storm. #Grammys #SecondhandEmbarassment','Can they not show Taylor swift when mumford is playing? #Grammys','Taylor swift is gods gift #fit #grammys',
                    'Please stop showing Taylor Swift on my tv screen. Im able to punch a hole through it. #GrammyAwards', 'Taylor Swift isnt even country, let alone anything else! #Justquit #Grammys','Not watching the Grammys because I know Tayloswift or lady gaga or Rihanna will win something and Ill be pissed off','Underwood, chris brown, wiz kalifa, taylor swift, i mean come on they all suck. The Grammys are a joke nowadays.','I really enjoyed the Illuminati Twitter handle until they tried to tell me Taylor Swift and Fun. are members. #Grammys','Why yall hating on Taylor Swift. She cares about the same amount of Grammy performances you have.','Taylor Swift just lost the last bit of respect I had for her ttt #Grammys','I just want to remind everybody that I started to listen to Mumford And Sons in seventh grade', 'Cant wait for Mumford #GRAMMYs','evon by Mumford and Sons, the black keys w Dr. John, jack white and NPH? #GRAMMYs done good this year. Couching it in Boca w my BK BFF','I want to see Mumford and Sons perform','I am SO excited that Mumford is playing tonight! #Grammys','Looking forward to Mumford and Sons! #grammy','Rocking my Mumford shirt right now. Obsessed. #GRAMMYS'])

y_train=[[0],[0],[0,1],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[0],[1],[1],[1],[1],[1],[1],[1]]
target_names = ['Taylor Swift', 'mumford & sons']
x_test= np.array(['Okay wtf dis TAYLOR FREAKEN SWIIFT SO AT THE GRAMMYS PEOPLE','Everyone bitching about how terrible T Swift is makes me glad Im not even watching the Grammys', 'Watching the Grammys - its clear that T-Swizzle is on drugs. Lots of drugs', 'Who is Tay going to go after these Grammy performances Bc she is due for a new boyfriend to break up with soon'])
classifier = Pipeline([('vectorizer', CountVectorizer()), ('tfidf', TfidfTransformer()), ('clf', OneVsRestClassifier(LinearSVC()))])
classifier.fit(x_train, y_train)
predicted = classifier.predict(x_test)
for items, labels in zip(x_test, predicted):
    print('%s => %s' % (items, ','.join(target_names[x] for x in labels)))
