import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from multiprocessing import Pool, cpu_count


def satterthwaite(dfs: list[float], vars: list[float]) -> float:
    """
    Calculate Degrees of Freedom (DF) using
    Satterthwaite approximation:

        [sum_i var ]²  / [ sum_i  var_i²/df_i  ]

    Parameters:
    -----------
    dfs: list[float]
        List of degrees of freedom
    vars: list[float]
        List of variances

    Returns:
    --------
    satter_df: float
        Degrees of Freedom
        calculated from satterhwaite
    """
    # Cast to numpy array
    vars = np.array(vars)
    dfs = np.array(dfs)

    # calculate numerator and denominator
    numerator_coef = np.sum(vars)
    denom_coef = (vars**2) / dfs

    satter_df = numerator_coef**2 / np.sum(denom_coef)

    return satter_df


def worker(pool_list: list):
    var1, df1, var2, df2 = pool_list
    return (satterthwaite(dfs=[df1, df2], vars=[var1, var2]), var1, df1, var2, df2)


if __name__ == "__main__":
    fixed_var = 1
    fixed_df = 10

    var = np.linspace(start=0.1, stop=4, num=1000)
    df = np.linspace(start=0.1, stop=20, num=20)

    pool_list = []

    for v in var:
        for d in df:
            pool_list.append([fixed_var, fixed_df, v, d])

    # Apply satterwaithe in multipool
    num_processes = cpu_count()
    with Pool(processes=num_processes) as pool:
        results = pool.map(worker, pool_list)

    df = pd.DataFrame(results, columns=["satter", "var1", "df1", "var2", "df2"])
    df["df_ratio"] = df["df2"] / df["df1"]

    fig = plt.figure()
    sns.lineplot(data=df, x="var2", y="satter", hue="df_ratio")
    plt.xlabel("Variance Ratio")
    plt.ylabel("DF")
    plt.title("Satterthwaite DF")
    plt.show()
