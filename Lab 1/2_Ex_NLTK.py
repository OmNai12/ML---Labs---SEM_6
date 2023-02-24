import nltk
from nltk.corpus import opinion_lexicon
import matplotlib.pyplot as plt
import random

# Importing the data set
nltk.download('opinion_lexicon')
# Fetching possitive and ngative reviews
opinion_lexicon.negative()
opinion_lexicon.positive()
# Printintg thme
print(len(opinion_lexicon.positive()))
print(len(opinion_lexicon.negative()))
# Printing the type
print(type(len(opinion_lexicon.negative())))
print(type(len(opinion_lexicon.positive())))
# Printing the plot
fig = plt.figure(figsize=(5, 5))
labels = 'Positive', 'Negative'
sizes = [len(opinion_lexicon.positive()), len(opinion_lexicon.negative())]
plt.pie(sizes, labels=labels, autopct='%.2f%%',
        shadow=True, startangle=90)
plt.axis('equal')
plt.show()
# Showing opinion colour wise
positive = opinion_lexicon.positive()
negative = opinion_lexicon.negative()
print('\033[92m' + positive[random.randint(0, 500)])
print('\033[91m' + negative[random.randint(0, 500)])
print(positive[26])
