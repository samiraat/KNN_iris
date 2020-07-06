import csv
import random
import math
import pandas as pd
import operator
import matplotlib.pyplot as plt
import seaborn as sns

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
	with open(filename, 'rb') as csvfile:
		lines = csv.reader(csvfile)
		dataset = list(lines)
		for x in range(len(dataset)):
			for y in range(4):
				dataset[x][y] = float(dataset[x][y])
			if random.random() < split:
				trainingSet.append(dataset[x])
			else:
				testSet.append(dataset[x])
				
def euclideanDistance(instance1, instance2, length):
	distance = 0
	for x in range(length):
		distance += pow((instance1[x] - instance2[x]), 2)
	return math.sqrt(distance)
	
def Neighbors(trainingSet, testInstance, k):
	distances = []
	length = len(testInstance)-1
	for x in range(len(trainingSet)):
		dist = euclideanDistance(testInstance, trainingSet[x], length)
		distances.append((trainingSet[x], dist))
	distances.sort(key = operator.itemgetter(1))
	neighbors = []
	for x in range(k):
		neighbors.append(distances[x][0])
	return neighbors
	
def predictclass(neighbors):
	# max vote class neighbors
	classVotes = {}
	for x in range(len(neighbors)):
		lable = neighbors[x][-1]
		if lable in classVotes:
			classVotes[lable] += 1
		else:
			classVotes[lable] = 1
	sortedVotes = sorted(classVotes.iteritems(), key=operator.itemgetter(1), reverse=True)
	return sortedVotes[0][0]
	
def Accuracy(testSet, predictions):
	correct = 0
	for x in range(len(testSet)):
		if testSet[x][-1] == predictions[x]:
			correct += 1
	return (correct/float(len(testSet))) * 100.0
				
def main():
	trainingSet=[]
	testSet=[]
	split = 0.70
	loadDataset('iris.data', split, trainingSet, testSet)
	print 'Train set: ' + repr(len(trainingSet))
	print 'Test set: ' + repr(len(testSet))  

	predictions=[]
	k = 3
	for x in range(len(testSet)):
		neighbors = Neighbors(trainingSet, testSet[x], k)
		#print testSet[x]
		#print neighbors
		result = predictclass(neighbors)
		predictions.append(result)
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy1 = Accuracy(testSet, predictions)
	print 'Accuracy: ', accuracy1
	k = 5
	predictions=[]
	for x in range(len(testSet)):
		neighbors = Neighbors(trainingSet, testSet[x], k)
		#print testSet[x]
		#print neighbors
		result = predictclass(neighbors)
		predictions.append(result)
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy2 = Accuracy(testSet, predictions)
	print 'Accuracy: ', accuracy2
	k = 7
	predictions=[]
	for x in range(len(testSet)):
		neighbors = Neighbors(trainingSet, testSet[x], k)
		#print testSet[x]
		#print neighbors
		result = predictclass(neighbors)
		predictions.append(result)
		#print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
	accuracy3 = Accuracy(testSet, predictions)
	print 'Accuracy: ', accuracy3
'''
####################plot
	log_cols = ["K nearest neighbor", "Accuracy"]
	log 	 = pd.DataFrame(columns=log_cols)

	names = ["K=3", "K=5", "K=7"]
	iris_scores = [accuracy1,accuracy2,accuracy3]

	for i in range(len(names)):
		log_entry = pd.DataFrame([[names[i], iris_scores[i]]], columns=log_cols)
		log = log.append(log_entry)
	sns.pointplot(y='Accuracy', x='K nearest neighbor', data=log, color="b")
	plt.show()
'''
main()