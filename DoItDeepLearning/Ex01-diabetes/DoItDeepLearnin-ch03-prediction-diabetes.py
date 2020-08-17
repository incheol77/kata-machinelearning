
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.datasets import load_breast_cancer

#############################
# prepare datasets
#############################
diabetes = load_diabetes()
x = diabetes.data[:,2]
y = diabetes.target
print(diabetes.DESCR)


#############################
# traing
#############################
w, b = 1.0, 1.0
w0, b0 = w, b
w1, b1 = 0, 0
epochs = 100
for i in range(1, epochs):
  for x_i, y_i in zip(x, y):
    y_hat = x_i * w + b
    err = y_i - y_hat
    w_rate = x_i
    w = w + w_rate * err
    b = b + 1 * err
    if i == 1:
      w1, b1 = w, b

print("w0, b0     : ", w0, b0)
print("w1, b1     : ", w1, b1)
print("w100, b100 : ", w, b)


#############################
# prediction
#############################
x_new = 0.18
y_pred_t1 = x_new * w1 + b1
y_pred_t100 = x_new * w + b
print("y_pred_t1, y_pred_t100 : ", y_pred_t1, y_pred_t100)


#############################
# display model
#############################
def make_point(w, b):
  start_x, end_x = -0.1, 0.15
  return ((start_x, start_x * w + b), (end_x, end_x * w + b))

# before training
pt1_0, pt2_0 = make_point(w0, b0)
plt.plot([pt1_0[0], pt2_0[0]], [pt1_0[1], pt2_0[1]])

# traing once
pt1_1, pt2_1 = make_point(w1, b1)
plt.plot([pt1_1[0], pt2_1[0]], [pt1_1[1], pt2_1[1]])

# traing 100 times
pt1_100, pt2_100 = make_point(w, b)
plt.plot([pt1_100[0], pt2_100[0]], [pt1_100[1], pt2_100[1]])

plt.scatter(x, y)
plt.scatter(x_new, y_pred_t0)
plt.scatter(x_new, y_pred_t1)
plt.scatter(x_new, y_pred_t100)
plt.xlabel('x')
plt.ylabel('y')
plt.show()
