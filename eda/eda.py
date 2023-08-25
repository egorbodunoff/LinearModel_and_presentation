import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import warnings
import os

warnings.simplefilter("ignore")


def open_data(path="../data/datasets"):
    datasets = os.listdir(path)
    datasets = [df.split('.')[0] for df in datasets]

    df_dict = {df: pd.read_csv(path + "/" + df + ".csv") for df in datasets}
    df_dict["D_clients"].rename(columns={"ID": "ID_CLIENT"}, inplace=True)
    return df_dict


def data_processor(df_dict):
    data = df_dict["D_clients"]
    data_loan = create_data_loan(df_dict)

    data = data.merge(df_dict["D_target"], on="ID_CLIENT", how="right")
    data = data.merge(df_dict["D_salary"], on="ID_CLIENT", how="left")
    data = data.merge(data_loan, on="ID_CLIENT", how="left")
    data = data[data["PERSONAL_INCOME"] > 100]
    data.drop(columns=["REG_ADDRESS_PROVINCE", "FACT_ADDRESS_PROVINCE",
                       "ID_CLIENT", "POSTAL_ADDRESS_PROVINCE"], inplace=True)
    return data


def create_data_loan(df_dict):
    data_loan = df_dict["D_loan"]

    data_loan = data_loan.merge(df_dict["D_close_loan"], on="ID_LOAN", how="inner")
    data_loan = data_loan.groupby("ID_CLIENT").agg({"ID_LOAN": "count", "CLOSED_FL": "sum"})
    data_loan.rename(columns={"ID_LOAN": "LOAN_NUM_TOTAL", "CLOSED_FL": "LOAN_NUM_CLOSED"}, inplace=True)

    return data_loan


def target_variable_exploration(df, target, positive=1):
    positive_class = df[df[target] == positive].shape[0]
    negative_class = df[target].shape[0] - positive_class
    positive_per = positive_class / df.shape[0] * 100
    negative_per = negative_class / df.shape[0] * 100

    fig = plt.figure(figsize=(4, 2))
    sns.countplot(x=df[target])
    plt.xlabel("class", size=6, labelpad=8)
    plt.ylabel("count", size=6, labelpad=8)
    plt.xticks([1, 0], [f"Positive {positive_per:.2f}%", f"Negative class {negative_per:.2f}%"])
    plt.tick_params(axis='x', labelsize=6)
    plt.tick_params(axis='y', labelsize=6)
    plt.title("Distribution of classes", size=8, y=1.05)

    return fig


def number_variable_exploration(data, features=("AGE", "PERSONAL_INCOME")):
    fig, axs = plt.subplots(figsize=(4, 2), ncols=2)
    for i, feature in enumerate(features):
        sns.histplot(data[feature].values, bins=40, kde=True, ax=axs[i])
        axs[i].set_xlabel(feature, size=6)
        axs[i].set_ylabel("count", size=6)
        axs[i].tick_params(labelsize=6)
    plt.tight_layout()

    return fig


def log_variable(data, feature="PERSONAL_INCOME"):
    fig = plt.figure(figsize=(4, 2))
    sns.histplot(np.log1p(data[feature].values), bins=50, kde=True)
    plt.xlabel(feature, size=6)
    plt.ylabel("count", size=6)
    plt.tick_params(labelsize=6)

    data[feature + "_LOG"] = np.log1p(data[feature].values)
    data.drop(columns=feature, axis=1, inplace=True)

    return fig, data


def dependence_on_target(data, features=("AGE", "PERSONAL_INCOME_LOG", "CHILD_TOTAL", "EDUCATION")):
    fig, axs = plt.subplots(figsize=(6, 3), ncols=4)
    for i, feature in enumerate(features):
        axs[i].scatter(data[feature], data["TARGET"], alpha=0.2)
        axs[i].set_xlabel(feature, size=5)
        axs[i].set_ylabel("target", size=5)
        axs[i].tick_params(labelsize=5)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    return fig


def corr(data, features=("AGE", "CHILD_TOTAL", "DEPENDANTS",
                         "PERSONAL_INCOME_LOG", "LOAN_NUM_TOTAL", "TARGET")):
    corrmat = data[list(features)].corr()
    fig = plt.figure(figsize=(5, 3))
    sns.heatmap(corrmat, square=True, annot=True)
    plt.title("Кореляционная матрица", fontsize=8)
    plt.tick_params(labelsize=6)

    return fig
