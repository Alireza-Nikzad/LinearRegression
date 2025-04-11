from pydataset import data
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split



df = data("Boston")

X = df[['rm']]
y = df[['medv']]

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

r2 = r2_score(y_test, y_pred)
print(r2)

plt.scatter(x_test, y_test, color='blue', label = "Actuall prices", alpha = 0.5)
plt.plot(x_test, y_pred, color= 'red', linewidth = 2, label = "prediction line")
plt.xlabel("Average number of rooms")
plt.ylabel("Median value of homes")
plt.title("Boston housing- Linear Regerssion")
plt.legend()
plt.show()