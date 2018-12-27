import csv
from sklearn.model_selection import train_test_split
import nltk

def main():
    stars = []
    reviews = []
    with open('Mexican_Restaurant_Reviews.csv', 'r', encoding='utf8') as csvfile:
        data = csv.reader(csvfile)
        next(data)
        for row in data:
            stars.append(int(row[3]))
            reviews.append(row[4])

    reviews_train, reviews_test, stars_train, stars_test = train_test_split(reviews, stars, test_size = 0.3, random_state = 42)

    trainset = []
    testset = []

    for i in range(0, len(reviews_train)):
        words = reviews_train[i].split()
        trainset.append((words, stars_train[i]))

    for i in range(0, len(reviews_test)):
        words = reviews_test[i].split()
        testset.append((words, stars_test[i]))

    trainset_formatted = [(list_to_dict(element[0]), element[1]) for element in trainset]
    testset_formatted = [(list_to_dict(element[0]), element[1]) for element in testset]

    algorithm = nltk.classify.MaxentClassifier.ALGORITHMS[0]
    classifier = nltk.MaxentClassifier.train(trainset_formatted, algorithm, max_iter=10)
    classifier.show_most_informative_features(10)

    count = 0

    for review in testset_formatted:
        label = review[1]
        text = review[0]
        determined_label = classifier.classify(text)
        if label == determined_label:
            count += 1

    print(count / len(reviews_test))


def list_to_dict(words_list):
  return dict([(word, True) for word in words_list])

main()