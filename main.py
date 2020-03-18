import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.linear_model as linear_model_classifier

gdp_label = 'GDP per capita'
life_satisfaction_label = 'Life satisfaction'


def prepare_country_stats(oecd_bli, gdp_per_capita):
    oecd_bli = oecd_bli[oecd_bli["INEQUALITY"]=="TOT"]
    oecd_bli = oecd_bli.pivot(index="Country", columns="Indicator", values="Value")
    gdp_per_capita.rename(columns={"2015": "GDP per capita"}, inplace=True)
    gdp_per_capita.set_index("Country", inplace=True)
    full_country_stats = pd.merge(left=oecd_bli, right=gdp_per_capita,
                                  left_index=True, right_index=True)
    full_country_stats.sort_values(by="GDP per capita", inplace=True)
    remove_indices = [0, 1, 6, 8, 33, 34, 35]
    keep_indices = list(set(range(36)) - set(remove_indices))
    return full_country_stats[["GDP per capita", 'Life satisfaction']].iloc[keep_indices]


def load_data():
    oecd_bli = pd.read_csv('data/oecd_bli_2015.csv', thousands=',')
    gdp_per_capita = pd.read_csv('data/gdp_per_capita.csv', thousands=',', delimiter='\t', encoding='latin1', na_values='n/a')
    return oecd_bli, gdp_per_capita


def plot_data(country_stats):
    country_stats.plot(kind='scatter', x=gdp_label, y=life_satisfaction_label)
    plt.show()


if __name__=='__main__':
    oecd_bli, gdp_per_capita = load_data()

    country_stats = prepare_country_stats(oecd_bli, gdp_per_capita)

    x = np.c_[country_stats[gdp_label]]
    y = np.c_[country_stats[life_satisfaction_label]]

    plot_data(country_stats)

    model = linear_model_classifier.LinearRegression()
    model.fit(x, y)

    input = [[22587]]
    print(model.predict(input))

