from keras.models import Sequential
from keras.layers import Dense
import numpy
from urllib import request
import matplotlib.pyplot as plt
%matplotlib inline 

# 데이터 세트의 URL을 설정
url = "https://gist.github.com/ktisha/c21e73a1bd1700294ef790c56c8aec1f"
f = request.urlopen(url)

# random seed for reproducibility
numpy.random.seed(2)

# 데이터 세트를 불러옵니다. 
f = open("sample_data/pima-indians-diabetes.csv")
dataset = numpy.loadtxt(f, delimiter=",", skiprows=10)


# split into input (X) and output (Y) variables, splitting csv data
X = dataset[:,0:8]
Y = dataset[:,8]



# 데이터세트를 두 가지 원인(X) 과 결과(Y)로 나누어 줍니다.
X = dataset[:,0:8]
Y = dataset[:,8]

# create model, add dense layers one by one specifying activation function
model = Sequential()
model.add(Dense(12, input_dim=8, activation='relu')) # input layer requires input_dim param
model.add(Dense(15, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid')) # sigmoid instead of relu for final probability between 0 and 1

# compile the model, adam gradient descent (optimized)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=['accuracy'])

# call the function to fit to the data (training the network)
# verbose=0 는 프로그래스를 숨깁니다.
history = model.fit(X, Y, epochs = 800, batch_size=10, verbose=0)

# 모델의 정확도를 계산합니다.
scores = model.evaluate(X, Y)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))



# Get the figure and the axes
fig, (ax0, ax1) = plt.subplots(nrows=1,ncols=2, sharey=False, figsize=(10, 5))

# 모델의 정확도를 그립니다.
ax0.plot(history.history['accuracy'])
ax0.set(title='model accuracy', xlabel='epoch', ylabel='accuracy')

# 모델의 오차를 그립니다.
ax1.plot(history.history['loss'])
ax1.set(title='model loss', xlabel='epoch', ylabel='loss')



# 가상의 환자 데이터 입력
patient_1 = numpy.array([[0,137,90,35,168,43.1,2.288,33]])

# 모델로 예측하기
prediction = model.predict(patient_1)

# 예측결과 출력하기
print("당뇨병에 걸릴 확률 : ", prediction*100)



