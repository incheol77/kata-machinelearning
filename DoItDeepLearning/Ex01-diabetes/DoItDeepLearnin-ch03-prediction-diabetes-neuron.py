import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

#############################
# Make neuron
#############################
class Neuron:
  def __init__(self):
    self.w = 1.0
    self.b = 1.0

  def forpass(self, x):
    y_hat = x * self.w + self.b
    return y_hat

  def backprop(self, x, err):
    w_grad = x * err
    b_grad = 1 * err
    return w_grad, b_grad

  def fit(self, x, y, epochs=100):
    for i in range(epochs):
      for x_i, y_i in zip(x, y):
        y_hat = self.forpass(x_i)
        err = -(y_i - y_hat)
        w_grad, b_grad = self.backprop(x_i, err)
        self.w -= w_grad
        self.b -= b_grad
      
  def predict(self, x):
    return x * self.w + self.b


#############################
# training & predicting
#############################
def make_point(w, b):
  start_x, end_x = -0.1, 0.15
  return ((start_x, start_x * w + b), (end_x, end_x * w + b))

x = diabetes.data[:,2] 
y = diabetes.target
neuron = Neuron()
pred_x = 0.18

# before training
y_pred_t0 = neuron.predict(pred_x)
pt1_0, pt2_0 = make_point(neuron.w, neuron.b)
plt.plot([pt1_0[0], pt2_0[0]], [pt1_0[1], pt2_0[1]])

# traing once
neuron.fit(x, y, 1)
y_pred_t1 = neuron.predict(pred_x)
pt1_1, pt2_1 = make_point(neuron.w, neuron.b)
plt.plot([pt1_1[0], pt2_1[0]], [pt1_1[1], pt2_1[1]])

# traing 100 times
neuron.fit(x, y, 100)
y_pred_t100 = neuron.predict(pred_x)
pt1_100, pt2_100 = make_point(neuron.w, neuron.b)
plt.plot([pt1_100[0], pt2_100[0]], [pt1_100[1], pt2_100[1]])


#############################
# display model
#############################
print("y_pred_t1, y_pred_t100 : ", y_pred_t1, y_pred_t100)
plt.scatter(x, y)
plt.scatter(pred_x, y_pred_t0)
plt.scatter(pred_x, y_pred_t1)
plt.scatter(pred_x, y_pred_t100)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
