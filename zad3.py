import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


def main():
    shopping_data = pd.DataFrame({
        'x': [5, 7, 10, 15, 20, 25],
        'y': [8, 10, 13, 18, 22, 25]
    })

    X = shopping_data.iloc[:, :-1]
    y = shopping_data.iloc[:, 1]

    plt.scatter(X, y, color='blue')
    plt.title('Престой на клиентите в магазина')
    plt.xlabel('Продължителност на престоя(минути)')
    plt.ylabel("Харчения в магазина(лева)")
    plt.show()

    model = LinearRegression()
    model.fit(X, y)

    print("Наклон на линията: ", model.coef_)
    print("Свободен член: ", model.intercept_)

    r2 = model.score(X, y)
    print(f'R^2 = {r2:.2f}')

    line_y = model.predict(X)
    plt.scatter(X, y, color="blue")
    plt.plot(X, line_y, color="red")
    plt.title("Престой на клиентите в магазина")
    plt.xlabel('Продължителност на престоя(минути)')
    plt.ylabel('Харчения в магазина (лева)')
    plt.show()

    new_X = [[12]]
    new_y = model.predict(new_X)
    print(f'За престой от 12 минути се очаква харченето да е {new_y[0]:.2f} лева')


main()
