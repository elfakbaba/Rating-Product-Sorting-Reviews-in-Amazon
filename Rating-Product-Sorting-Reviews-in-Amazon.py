import pandas as pd
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler


pd.set_option("display.max_columns", None)
pd.set_option('display.max_rows', None)
pd.set_option("display.width", 200)

df= pd.read_csv("datasets/amazon_review.csv")
df.head()

# Calculating Average Rating Based on Current Reviews

# Average Rate
df["overall"].mean() #4.587589013224822

df.describe().T


#Time-based weighting
def time_based_weighted_average(dataframe, w1=40, w2=30, w3=20, w4=10):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100

time_based_weighted_average((df)) #4.628116998159475


###################################################
# Specifying 20 Reviews to Display on the Product Detail Page for the Product
###################################################

df["helpful_no"] = df["total_vote"]- df["helpful_yes"]
df.sort_values("total_vote", ascending=False).head(12)

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score

     - The lower limit of the confidence interval to be calculated for the Bernoulli parameter p is accepted as the WLB score.
     - The score to be calculated is used for product ranking.
     - Note:
     If the scores are between 1-5, 1-3 are marked as negative, 4-5 as positive and can be made to conform to Bernoulli.
     This brings with it some problems. For this reason, it is necessary to make a bayesian average rating.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)
df.head()


def score_pos_neg_diff(df, up, down):
    df["score_pos_neg_diff"] = df[up] - df[down]

score_pos_neg_diff(df,"helpful_yes","helpful_no")

def score_average_rating (df,up,all):
    df["score_average_rating"] = df[up]/ df[all]

score_average_rating(df,"helpful_yes","total_vote")


df.sort_values("score_pos_neg_diff", ascending=False).head(20)

df.sort_values("score_average_rating", ascending=False).head(20)

#*
df.sort_values("wilson_lower_bound", ascending=False).head(20)