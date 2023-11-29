import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer.factor_analyzer import FactorAnalyzer

def find_factors(X: pd.DataFrame, fa : FactorAnalyzer):
    fa.fit(X)
    ev, v = fa.get_eigenvalues()
    plt.scatter(range(1, X.shape[1]+1), ev)
    plt.plot(range(1, X.shape[1]+1), ev)
    plt.plot(range(1, X.shape[1]+1), np.ones(shape=(len(X.columns),)))
    plt.title('Scree Plot')
    plt.xlabel('Factors')
    plt.ylabel('Eigenvalue')
    plt.grid(True)
    plt.show()
    return ev

def perform_fa(X : pd.DataFrame, fa : FactorAnalyzer):
    fa.fit(X)
    factor_loadings = pd.DataFrame(fa.loadings_, columns=['Factor'+str(x) for x in range(1,fa.n_factors+1)], index=[X.columns])
    factor_vars = pd.DataFrame(fa.get_factor_variance(), index=["Variance", "%Var", "Cum. Var."], columns = ['Factor'+str(x) for x in range(1,fa.n_factors+1)])
    return pd.concat([factor_loadings, factor_vars], ignore_index=False, axis=0)

def plot_fa_matrix(fa : FactorAnalyzer, X : pd.DataFrame, figsize : tuple = (15,15)):
    fa.fit(X)
    factor_loadings = pd.DataFrame(fa.loadings_, columns=['Factor'+str(x) for x in range(1,fa.n_factors+1)], index=[X.columns])
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_title("Factor loadings of FA")
    g = sns.heatmap(factor_loadings, center = 0, square=True, linewidths=.5, cbar=True, vmin=-1, vmax=1, annot=True, ax = ax, fmt=".2f")
    return g

def show_factor_to_column(fa : FactorAnalyzer, X : pd.DataFrame):
    fa.fit(X)
    factor_loadings = pd.DataFrame(fa.loadings_, columns=['Factor'+str(x) for x in range(1,fa.n_factors+1)], index=[X.columns])
    dic = {}
    for row in factor_loadings.iterrows():
        title = row[0][0]
        vals = row[1]
        x = list(vals).index(vals.max()) + 1
        #print(str(title) + ":" + str(x))
        dic_val = dic.get(x)
        if dic_val:
            dic_val.append(title)
            dic.update({x:dic_val})
        else:
            dic.update({x:[title]})
    arr = []
    for name in dic.keys():
        arr.append([name, dic.get(name), len(dic.get(name))])
    return pd.DataFrame(arr, columns=["Factor","Columns","Length"]).set_index("Factor").sort_index()