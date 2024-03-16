import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

df = pd.read_csv('zad4_b.csv', sep=';', index_col=0)

print(df)


X = df.columns.astype(int).values.reshape(-1, 1)
y = df.iloc[0].astype(int).values

model = LinearRegression()
model.fit(X, y)

print("Наклон на линията: ", model.coef_)
print("Свободен член: ", model.intercept_)

r2 = model.score(X, y)
print(f'R^2 = {r2:.2f}')

area_to_predict = [[75]]
predicted_price = model.predict(area_to_predict)
print("Очаквана цена при 75 квадрата:", predicted_price[0])

plt.figure(figsize=(8, 6))
plt.scatter(X, y, color='blue', label='Цена')
plt.plot(X, model.predict(X), color='red', label='Линейна Регресия')
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Цени на имоти')
plt.legend()
plt.grid(True)
plt.show()
