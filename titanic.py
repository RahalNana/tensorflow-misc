import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

DATADIR = "E:/Pycharm Projects/Datasets/titanic/"

train_df = pd.read_csv(DATADIR + "train.csv")
train_df.set_index("PassengerId", inplace=True)

df_num = train_df[["Age", "SibSp", "Parch", "Fare"]]
df_cat = train_df[["Survived", "Pclass", "Sex", "Ticket", "Cabin", "Embarked"]]

######## Plotting Histograms of data
# for col in df_num.columns:
#     plt.figure()
#     plt.hist(df_num[col])
#     plt.title(col)
# plt.show()

######## Plotting scatter plots of data
# xAx = "Fare"
# yAx = "Age"
#
# df_survived = train_df[df_cat["Survived"] > 0]
# df_not_survived = train_df[df_cat["Survived"] == 0]
# survived = plt.scatter(df_survived[xAx], df_survived[yAx])
# not_survived = plt.scatter(df_not_survived[xAx], df_not_survived[yAx])
# plt.xlabel(xAx)
# plt.ylabel(yAx)
# plt.legend((survived, not_survived), ("Survived", "Not Survived"))
# plt.show()

for col in df_cat.columns:
    plt.figure()
    sns.barplot(df_cat[col].value_counts().index, df_cat[col].value_counts()).set_title(col)
    plt.show()
