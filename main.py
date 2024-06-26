# importing libraries
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

data = pd.read_csv("./Food_Production.csv")

print(data.columns)

# model 1
X1= data[['Farm', 'Processing', 'Transport']]
Y1= data['Total_emissions']

model1 = LinearRegression()

X_train1, X_test1, Y_train1, Y_test1 = train_test_split(X1, Y1, test_size=0.2, random_state=50)

model1.fit(X_train1, Y_train1)

model_1_r_squared = model1.score(X_test1, Y_test1)

print('Model 1 R-squared on test data:', model_1_r_squared)
# -> Model 1 R-squared on test data: 0.8957300065577624

# model 2
X2= data[['Retail', 'Animal Feed', 'Transport']]
Y2= data['Total_emissions']

model2 = LinearRegression()

X_train2, X_test2, Y_train2, Y_test2 = train_test_split(X2, Y2, test_size=0.2, random_state=50)

model2.fit(X_train2, Y_train2)

model_2_r_squared = model2.score(X_test2, Y_test2)
print('Model 2 R-squared on test data:', model_2_r_squared)

# -> Model 2 R-squared on test data: 0.8372989874331487

# model 3
X3= data[['Processing', 'Retail', 'Transport']]
Y3= data['Total_emissions']

model3 = LinearRegression()

X_train3, X_test3, Y_train3, Y_test3 = train_test_split(X3, Y3, test_size=0.2, random_state=50)

model2.fit(X_train3, Y_train3)

model_3_r_squared = model2.score(X_test3, Y_test3)

print('Model 3 R-squared on test data:', model_3_r_squared)
# -> Model 3 R-squared on test data: -0.2685101545174742

# Predicting from Model 1 because that is our best model

y_predicted = model1.predict(X_test1)


# Predicted Values
print('=======================================')
coefficients = model1.coef_
print('The Coefficients are:', coefficients) # The Coefficients are: [1.38709946 3.37106976 1.21181311]
intercept = model1.intercept_ 
print('The Intercept is:', intercept) # The Intercept is: 0.44836422727058434

model_mse = mean_squared_error(Y_test1, y_predicted) 
print('The Mean Squared Error is:', model_mse) # The Mean Squared Error is: 9.405256391199405
model_r2_score = r2_score(Y_test1, y_predicted)
print('The R2 is:', model_r2_score) # The R2 is: 0.8957300065577624
print('=======================================')

plt.figure(figsize=(8, 6))
plt.scatter(X_test1['Farm'], Y_test1, color='r', label='Farm')
plt.scatter(X_test1['Processing'], Y_test1, color='g', label='Processing')
plt.scatter(X_test1['Transport'], Y_test1, color='b', label='Transport')
# plt.plot(Y_test1, y_predicted, color='y', label='Best Fit Line')
plt.title('Linear Regression')
plt.legend()
plt.show()