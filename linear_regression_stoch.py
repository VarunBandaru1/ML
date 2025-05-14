import pandas as pd
import matplotlib.pyplot as plt
import random


learning_rate = 0.01
iterations = 100000
training_split = 50

data_frame = pd.read_csv("Iris.csv")
data_frame = data_frame[data_frame["Species"] ==  "Iris-setosa"]
data_frame = data_frame.drop(columns = ["Id"])

plt.scatter(data_frame["SepalLengthCm"] , data_frame["SepalWidthCm"])

total_data = [[[1 , data_frame["SepalLengthCm"].values.tolist()[i]], data_frame["SepalWidthCm"].values.tolist()[i]]for i in range(len(data_frame))]
training_data = random.sample(total_data, training_split)
testing_data = [data for data in total_data if data not in training_data]


def gradientStochDescent(hypothesis, current):
    return [current[0][i] * (current[1] - dotProduct(hypothesis , current[0])) for i in range(len(hypothesis))]

def trainData(iterations, training_data, learning_rate):
    hypothesis = [0,0]
    for i in range(iterations):
        for current in training_data:
            gradient = gradientStochDescent(hypothesis, current)
            hypothesis = [hypothesis[i] + gradient[i] * learning_rate for i in range(len(gradient))]
    return hypothesis


def dotProduct(hypothesis, input):
    result = 0
    for i in range(len(input)):
        result += hypothesis[i] * input[i]
    return result


linReg = trainData(iterations, training_data, learning_rate)
model = linReg
plt.plot([i for i in range(8)] , [dotProduct(model , [1 , i]) for i in range(8)])
plt.show()