import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def main():
    realestate_data = pd.DataFrame({
        'x': [45, 60, 65, 70, 80, 100],
        'y': [50000, 80000, 92000, 99000, 110000, 160000]

    })

    X = realestate_data.iloc[:, :-1]
    y = realestate_data.iloc[:, 1]

    model = LinearRegression()
    model.fit(X, y)

    print("Наклон на линията: ", model.coef_)
    print("Свободен член: ", model.intercept_)

    r2 = model.score(X, y)
    print(f'R^2 = {r2:.2f}')

    line_y = model.predict(X)
    plt.scatter(X, y, color="blue")
    plt.plot(X, line_y, color="red")
    plt.title("Цени на имоти")
    plt.xlabel('Квадратура')
    plt.ylabel('Цена')
    plt.show()

    new_X = [[75]]
    new_y = model.predict(new_X)
    print(f'За имот с 75 квадрата се очаква цената му да е {new_y[0]:.2f} лева')

main()
