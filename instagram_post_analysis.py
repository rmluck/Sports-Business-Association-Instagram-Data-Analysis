import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
import statsmodels
from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor

data = pd.read_csv("sba_instagram_posts.csv", encoding="latin1")
Data = pd.DataFrame(data)


def performance_analysis():
    total_impressions = Data["Impressions"].sum()
    total_likes = Data["Likes"].sum()
    total_saves = Data["Saves"].sum()
    total_comments = Data["Comments"].sum()
    total_shares = Data["Shares"].sum()

    engagement_rate = (total_likes + total_saves + total_comments + total_shares) / total_impressions
    print("Engagement Rate is calculated as: ", engagement_rate)

    print("Total impressions: ", total_impressions)
    print("Total likes: ", total_likes)
    print("Total saves: ", total_saves)
    print("Total comments: ", total_comments)
    print("Total shares: ", total_shares)


def source_analysis():
    def from_home():
        plt.figure(figsize=(10, 8))
        plt.style.use("fivethirtyeight")
        plt.title("Distribution of Impressions From Home")
        sns.distplot(data["Impressions From Home"])
        plt.show()

    def from_profile():
        plt.figure(figsize=(10, 8))
        plt.style.use("fivethirtyeight")
        plt.title("Distribution of Impressions From Profile")
        sns.distplot(data["Impressions From Profile"])
        plt.show()

    def from_hashtags():
        plt.figure(figsize=(10, 8))
        plt.style.use("fivethirtyeight")
        plt.title("Distribution of Impressions From Hashtags")
        sns.distplot(data["Impressions From Hashtags"])
        plt.show()

    def from_location():
        plt.figure(figsize=(10, 8))
        plt.style.use("fivethirtyeight")
        plt.title("Distribution of Impressions From Location")
        sns.distplot(data["Impressions From Location"])
        plt.show()

    def from_explore():
        plt.figure(figsize=(10, 8))
        plt.style.use("fivethirtyeight")
        plt.title("Distribution of Impressions From Explore")
        sns.distplot(data["Impressions From Explore"])
        plt.show()

    def from_other():
        plt.figure(figsize=(10, 8))
        plt.style.use("fivethirtyeight")
        plt.title("Distribution of Impressions From Other")
        sns.distplot(data["Impressions From Other"])
        plt.show()

    home = Data["Impressions From Home"].sum()
    profile = Data["Impressions From Profile"].sum()
    hashtags = Data["Impressions From Hashtags"].sum()
    location = Data["Impressions From Location"].sum()
    explore = Data["Impressions From Explore"].sum()
    other = Data["Impressions From Other"].sum()

    labels = ["From Home", "From Profile", "From Hashtags",
              "From Location", "From Explore", "From Other"]
    values = [home, profile, hashtags, location, explore, other]
    fig = px.pie(data, values=values, names=labels, title="Impressions on Instagram Posts From Various Sources",
                 hole=0.5)
    fig.show()


def user_behavior_analysis():
    profile_visits = Data["Profile Visits"].sum()
    follows = Data["Follows"].sum()
    likes = Data["Likes"].sum()

    print("Total profile visits from a post: ", profile_visits)
    print("Total follows from a post: ", follows)
    print("Total likes: ", likes)

    plt.figure(figsize=(8, 6))
    plt.bar(["Profile Visits", "Follows", "Likes"], [profile_visits, follows, likes])
    plt.title("User Behavior Analysis")
    plt.ylabel("Count")
    plt.show()


def analyze_content():
    def post_types():
        w = []
        for word in Data["Post Type"]:
            w.append(word)
        words = " ".join(w)
        stopwords = set(STOPWORDS)
        wc = WordCloud(stopwords=stopwords, background_color="white").generate(words)
        plt.figure(figsize=(12, 10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def quarter():
        w = []
        for word in Data["Quarter"]:
            w.append(word)
        words = " ".join(w)
        stopwords = set(STOPWORDS)
        wc = WordCloud(stopwords=stopwords, background_color="white").generate(words)
        plt.figure(figsize=(12, 10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def content():
        w = []
        for word in Data["Content"]:
            w.append(word)
        words = " ".join(w)
        stopwords = set(STOPWORDS)
        wc = WordCloud(stopwords=stopwords, background_color="white").generate(words)
        plt.figure(figsize=(12, 10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    def hashtags():
        w = []
        for word in Data["Hashtags"]:
            w.append(word)
        words = " ".join(w)
        stopwords = set(STOPWORDS)
        wc = WordCloud(stopwords=stopwords, background_color="white").generate(words)
        plt.figure(figsize=(12, 10))
        plt.imshow(wc, interpolation="bilinear")
        plt.axis("off")
        plt.show()

    post_types()


def analyze_relationships():
    def likes():
        figure = px.scatter(data_frame = data, x="Impressions", y="Likes", size="Likes",
                            trendline="ols", title="Relationship Between Likes and Impressions")
        figure.show()

    def comments():
        figure = px.scatter(data_frame = data, x="Impressions", y="Comments", size="Comments",
                            trendline="ols", title="Relationship Between Comments and Impressions")
        figure.show()

    def shares():
        figure = px.scatter(data_frame=data, x="Impressions", y="Shares", size="Shares",
                            trendline="ols", title="Relationship Between Shares and Impressions")
        figure.show()

    def saves():
        figure = px.scatter(data_frame=data, x="Impressions", y="Saves", size="Saves",
                            trendline="ols", title="Relationship Between Saves and Impressions")
        figure.show()

    correlation = Data.corr()
    print(correlation["Impressions"].sort_values(ascending=False))


def analyze_conversion_rate():
    conversion_rate = (Data["Follows"].sum() / Data["Profile Visits"].sum()) * 100
    print("Conversion rate: ", conversion_rate)

    def new_followers():
        figure = px.scatter(data_frame=data, x="Profile Visits", y="Follows", size="Follows", trendline="ols",
                            title = "Relationship Between Profile Visits and Followers Gained")
        figure.show()

    new_followers()


def reach_prediction_model():
    x = np.array(Data[["Likes", "Saves", "Comments", "Shares", "Profile Visits", "Follows"]])
    y = np.array(Data["Impressions"])
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 42)
    model = PassiveAggressiveRegressor()
    model.fit(x, y)
    print(model.score(x_test, y_test))

    features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
    print(model.predict(features))


reach_prediction_model()
