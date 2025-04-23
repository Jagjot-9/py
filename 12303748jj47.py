# Required Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from scipy.stats import ttest_ind, f_oneway
import ast
import warnings
warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("pyyyy.csv")

# ==============================
# 1. DATA CLEANING
# ==============================

# Drop rows with mostly missing values
df = df.dropna(thresh=5)

# Convert date columns to datetime
df['publishDate'] = pd.to_datetime(df['publishDate'], errors='coerce')
df['firstPublishDate'] = pd.to_datetime(df['firstPublishDate'], errors='coerce')

# Convert genres, characters, ratingsByStars, setting from string to lists
def parse_list_column(col):
    return col.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') else [])

list_cols = ['genres', 'characters', 'ratingsByStars', 'setting']
for col in list_cols:
    df[col] = parse_list_column(df[col])

# Fill missing price with median
df['price'] = df['price'].fillna(df['price'].median())

# Drop rows with missing ratings or pages
df = df.dropna(subset=['rating', 'pages'])

# ==============================
# 2. VISUALIZATIONS
# ==============================

# Explode the genres list column
df_genres = df.explode('genres')

# Count frequency of each genre
genre_counts = df_genres['genres'].value_counts()

# Get top 5 genres for focus
top_genres = genre_counts.head(5)
plt.figure(figsize=(8, 5))
sns.barplot(x=top_genres.index, y=top_genres.values, palette='viridis')
plt.title("Top 5 Most Read Genres")
plt.ylabel("Number of Books")
plt.xlabel("Genre")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Pie chart of top 5 genres
plt.figure(figsize=(7, 7))
plt.pie(top_genres.values, labels=top_genres.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette('pastel'))
plt.title("Most Read Genres Distribution")
plt.axis('equal')  # Equal aspect ratio ensures pie is drawn as a circle.
plt.tight_layout()
plt.show()


# Rating Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['rating'], bins=20, kde=True)
plt.title("Distribution of Ratings")
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

# Ratings by Top 5 Genres
df_genre_expanded = df.explode('genres')
top_genres = df_genre_expanded['genres'].value_counts().nlargest(5).index
top_df = df_genre_expanded[df_genre_expanded['genres'].isin(top_genres)]
plt.figure(figsize=(10, 6))
sns.boxplot(x='genres', y='rating', data=top_df)
plt.title("Rating Distribution by Top Genres")
plt.show()

# Publications over time
plt.figure(figsize=(10, 5))
df['year'] = df['publishDate'].dt.year
sns.countplot(x='year', data=df.sort_values('year'))
plt.xticks(rotation=90)
plt.title("Number of Books Published per Year")
plt.show()

# Author popularity
top_authors = df['author'].value_counts().nlargest(10)
plt.figure(figsize=(10, 6))
top_authors.plot(kind='bar')
plt.title("Top 10 Authors by Number of Books")
plt.ylabel("Book Count")
plt.xticks(rotation=45)
plt.show()

# Price vs Rating
plt.figure(figsize=(8, 6))
sns.scatterplot(x='rating', y='price', data=df)
plt.title("Rating vs Price")
plt.show()

# WordCloud from Descriptions
text = " ".join(df['description'].dropna().astype(str).values)
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud from Descriptions")
plt.show()

# ==============================
# 3. EXPLORATORY DATA ANALYSIS
# ==============================

print("Top 10 Genres:")
print(df_genre_expanded['genres'].value_counts().head(10))

print("Top 5 Publishers:")
print(df['publisher'].value_counts().head())

# Pages vs Rating
plt.figure(figsize=(8, 6))
sns.scatterplot(x='pages', y='rating', data=df)
plt.title("Pages vs Rating")
plt.show()

# ==============================
# 4. STATISTICAL ANALYSIS
# ==============================

# Correlation Matrix
plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# T-test: Do books with awards have higher ratings?
df['has_award'] = df['awards'].apply(lambda x: isinstance(x, str) and len(x) > 4)
t_stat, p_val = ttest_ind(df[df['has_award']]['rating'], df[~df['has_award']]['rating'])
print("T-Test: Awards vs No Awards")
print(f"T-statistic: {t_stat:.3f}, P-value: {p_val:.3f}")

# ANOVA: Rating across top genres
groups = [group['rating'].values for name, group in top_df.groupby('genres')]
f_stat, p_val = f_oneway(*groups)
print("ANOVA: Rating by Genre")
print(f"F-statistic: {f_stat:.3f}, P-value: {p_val:.3f}")

# Regression: Predict Price
features = ['rating', 'pages', 'likedPercent', 'numRatings']
df_model = df.dropna(subset=features + ['price'])
X = df_model[features]
y = df_model['price']
reg = LinearRegression().fit(X, y)
print("\nLinear Regression Coefficients (Price Prediction):")
for f, coef in zip(features, reg.coef_):
    print(f"{f}: {coef:.2f}")

# ==============================
# 5. CREATIVE IDEAS (Stub)
# ==============================

# Basic Recommendation (same genre & similar rating)
def recommend_books(book_title, n=5):
    book = df[df['title'].str.lower() == book_title.lower()]
    if book.empty:
        return "Book not found."
    book_genres = book.iloc[0]['genres']
    book_rating = book.iloc[0]['rating']
    similar = df[df['genres'].apply(lambda g: any(genre in g for genre in book_genres))]
    similar = similar[similar['title'] != book.iloc[0]['title']]
    similar['score'] = abs(similar['rating'] - book_rating)
    return similar.sort_values('score').head(n)[['title', 'author', 'rating']]

print("\nExample Recommendations:")
print(recommend_books("The Hunger Games"))
