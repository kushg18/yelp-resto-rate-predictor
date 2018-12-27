import json
import csv
import wordcloud
import numpy as np
import matplotlib.pyplot as plt
from random import shuffle
from os import path
from PIL import Image
from sklearn.model_selection import train_test_split

# for doc2Vec
from gensim import utils
from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
import re


# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.cross_validation import train_test_split

class LabeledLineSentence(object):
    def __init__(self, sources):
        self.sources = sources

        flipped = {}

        # make sure that keys are unique
        for key, value in sources.items():
            if value not in flipped:
                flipped[value] = [key]
            else:
                raise Exception('Non-unique prefix encountered')

    def __iter__(self):
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    yield LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no])

    def to_array(self):
        self.sentences = []
        for source, prefix in self.sources.items():
            with utils.smart_open(source) as fin:
                for item_no, line in enumerate(fin):
                    self.sentences.append(LabeledSentence(utils.to_unicode(line).split(), [prefix + '_%s' % item_no]))
        return self.sentences

    def sentences_perm(self):
        shuffle(self.sentences)
        return self.sentences


class Yelp:
    def __init__(self):
        self.businesses = {}
        self.reviews = {}

    def readBusiness(self):
        bCount = 0
        businessFile = "/Users/kushalgevaria/Desktop/RIT/Semester2/BDA720/Project/Dataset/dataset/business.json"
        with open(businessFile, 'r') as bFile:
            for bLine in bFile:
                bCount += 1
                business = json.loads(bLine)
                for key, value in business.items():
                    if key == "categories":
                        if "Mexican" in value:
                            self.businesses[business["business_id"]] = business["name"]
        bFile.close()
        print("Total Mexican Restaurants : ", len(self.businesses))
        print("Total Businesses : ", bCount)

    def readReviews(self):
        reviewFile = "/Users/kushalgevaria/Desktop/RIT/Semester2/BDA720/Project/Dataset/dataset/review.json"
        rCount = 0
        for id in self.businesses.keys():
            self.reviews[id] = []
        # print(reviews)
        with open('Mexican_Restaurant_Reviews.csv', 'w') as csvWrite:
            header = ["business_id", "name", "review_id", "stars", "text", "useful", "funny", "cool"]
            csvWriter = csv.writer(csvWrite)
            csvWriter.writerow(header)
            with open(reviewFile, 'r') as rFile:
                for rLine in rFile:
                    rCount += 1
                    review = json.loads(rLine)
                    # print(review)
                    bID = review["business_id"]
                    if bID in self.reviews.keys():
                        instanceList = [bID, self.businesses[bID], review["review_id"], review["stars"], review["text"],
                                        review["useful"], review["funny"], review["cool"]]
                        csvWriter.writerow(instanceList)
                        # print(bID)
                        self.reviews[bID].append(
                            {"review_id": review["review_id"], "stars": review["stars"], "text": review["text"],
                             "useful": review["useful"], "funny": review["funny"], "cool": review["cool"]})
            rFile.close()
        csvWrite.close()

    def extractJustReviews(self):
        print("Creating a text file containing all just all reviews.")
        with open('justReviews.txt', 'w') as writeReviews:
            with open('Mexican_Restaurant_Reviews.csv', 'r') as csvRead:
                readRows = csv.DictReader(csvRead, delimiter=',')
                for row in readRows:
                    writeReviews.write(row["text"] + "\n")
            csvRead.close()
        writeReviews.close()
        print("Created a text file named justReviews.txt")

    def wordCloud(self):
        print("Creating Word Cloud")
        d = path.dirname(__file__)
        allReviews = open(path.join(d, 'justReviews.txt')).read()
        yelpImage = np.array(Image.open(path.join(d, "Yelp_trademark_RGB.png")))
        stopwords = set(wordcloud.STOPWORDS)
        stopwords.add("english")  # to get rid of the most common words like "the", "it", "of" etc
        wc = wordcloud.WordCloud(background_color="white", max_words=2000, mask=yelpImage, max_font_size=40,
                                 stopwords=stopwords, random_state=42)
        wc.generate(allReviews)
        yelp_logo_colors = wordcloud.ImageColorGenerator(yelpImage)
        plt.imshow(wc.recolor(color_func=yelp_logo_colors), interpolation="bilinear")
        plt.axis("off")
        # plt.figure()
        plt.show()
        print("Created Word Cloud")

    def getTrainingAndTestingSets(self):
        stars = []
        reviews = []
        with open('Mexican_Restaurant_Reviews.csv', 'r', encoding='utf8') as csvfile:
            data = csv.reader(csvfile)
            next(data)
            for row in data:
                stars.append(int(row[3]))
                reviews.append(re.sub('[^A-Za-z0-9]+', ' ', row[4]))
        return train_test_split(reviews, stars, test_size=0.3, random_state=42)

    def createFilesForDoc2Vec(self, reviews_train, reviews_test, stars_train, stars_test):
        # positive training dataset
        with open('positive_train.txt', 'w') as writePosTrain:
            for index in range(len(stars_train)):
                if stars_train[index] == 5 or stars_train[index] == 4:
                    # print(reviews_train[index], stars_train[index])
                    writePosTrain.write(reviews_train[index] + "\n")
        writePosTrain.close()

        # neutral training dataset
        with open('neutral_train.txt', 'w') as writeNeuTrain:
            for index in range(len(stars_train)):
                if stars_train[index] == 3:
                    # print(reviews_train[index], stars_train[index])
                    writeNeuTrain.write(reviews_train[index] + "\n")
        writeNeuTrain.close()

        # negative training dataset
        with open('negative_train.txt', 'w') as writeNegTrain:
            for index in range(len(stars_train)):
                if stars_train[index] == 2 or stars_train[index] == 1:
                    # print(reviews_train[index], stars_train[index])
                    writeNegTrain.write(reviews_train[index] + "\n")
        writeNegTrain.close()

        # positive testing dataset
        with open('positive_test.txt', 'w') as writePosTest:
            for index in range(len(stars_test)):
                if stars_test[index] == 5 or stars_test[index] == 4:
                    writePosTest.write(reviews_test[index] + "\n")
        writePosTest.close()

        # neutral testing dataset
        with open('neutral_test.txt', 'w') as writeNeuTest:
            for index in range(len(stars_test)):
                if stars_test[index] == 3:
                    writeNeuTest.write(reviews_test[index] + "\n")
        writeNeuTest.close()

        # negative testing dataset
        with open('negative_test.txt', 'w') as writeNegTest:
            for index in range(len(stars_test)):
                if stars_test[index] == 2 or stars_test[index] == 1:
                    writeNegTest.write(reviews_test[index] + "\n")
        writeNegTest.close()

        # all reviews unlabelled
        with open('all_reviews.txt', 'w') as writeAllRev:
            for index in range(len(stars_train)):
                writeAllRev.write(reviews_train[index] + "\n")
            for index in range(len(stars_test)):
                writeAllRev.write(reviews_test[index] + "\n")
        writeAllRev.close()

    def doc2Vec(self):
        totalReviewCount = 0
        with open('all_reviews.txt', 'r') as readAllRev:
            totalReviewCount = len(readAllRev.readlines())
        print("Total Reviews -> ", totalReviewCount)

        sources = {'negative_test.txt': 'TEST_NEG', 'positive_test.txt': 'TEST_POS', 'neutral_test.txt': 'TEST_NEU',
                   'negative_train.txt': 'TRAIN_NEG',
                   'positive_train.txt': 'TRAIN_POS', 'neutral_train.txt': 'TRAIN_NEU', 'all_reviews.txt': 'TRAIN_UNS'}
        sentences = LabeledLineSentence(sources)
        print(sentences)
        model = Doc2Vec(min_count=1, window=10, size=100, sample=1e-4, negative=5, workers=8)
        model.build_vocab(sentences.to_array())
        for epoch in range(10):
            model.train(sentences.sentences_perm(), total_examples=totalReviewCount, epochs=model.iter)
        model.save('yelpDoc2Vec.d2v')
        print("Done")

        pass


def main():
    yelpObj = Yelp()
    # yelpObj.readBusiness()
    # yelpObj.readReviews()
    # yelpObj.extractJustReviews()
    # yelpObj.wordCloud()
    # reviews_train, reviews_test, stars_train, stars_test = yelpObj.getTrainingAndTestingSets()
    # yelpObj.createFilesForDoc2Vec(reviews_train, reviews_test, stars_train, stars_test)
    # yelpObj.doc2Vec()


if __name__ == '__main__':
    main()
