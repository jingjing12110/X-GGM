import seaborn as sns

sns.set()
import matplotlib.pyplot as plt
import pandas as pd


def plot_tail(alpha, accuracy, model_name='default'):
    data = {'Tail size': alpha, model_name: accuracy}
    df = pd.DataFrame(data, dtype=float)
    df = pd.melt(df, ['Tail size'], var_name="Models", value_name="Accuracy")
    ax = sns.lineplot(x="Tail size", y="Accuracy", hue="Models", style="Models",
                      data=df, markers=False, ci=None)
    plt.xscale('log')
    plt.ylim(0, 100)
    plt.savefig('./tail_plot_%s.pdf' % model_name)
    plt.close()


"""
Simple demo of a horizontal bar chart.
"""
import matplotlib.pyplot as plt
plt.rcdefaults()
import numpy as np
import matplotlib.pyplot as plt

# Example data
people = ('Tom', 'Dick', 'Harry', 'Slim', 'Jim')
y_pos = np.arange(len(people))
performance = 3 + 10 * np.random.rand(len(people))
error = np.random.rand(len(people))

plt.barh(y_pos, performance, xerr=error, align='center', alpha=0.4)
plt.yticks(y_pos, people)
plt.xlabel('Performance')
plt.title('How fast do you want to go today?')
plt.savefig("barh.eps",format="eps")