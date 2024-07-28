# Movie Recommendation System

This project implements a basic movie recommendation system using Python and Pandas. The system suggests movies similar to the ones users like, based on a dataset of movie information.

## Table of Contents

- [Overview](#overview)
- [Data](#data)
- [Techniques Used](#techniques-used)
- [Setup](#setup)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Overview

A recommender system is a tool that suggests products, services, or content to users based on various factors, such as past behavior or item characteristics. This project focuses on building a movie recommendation system using two common techniques:

1. **Collaborative Filtering**: Recommends items by identifying similar users or items based on user interactions.
2. **Content-Based Filtering**: Recommends items by comparing the attributes of items and suggesting similar ones based on user preferences.

## Data

The dataset used in this project contains information about movies, including:
- Movie ID
- Movie Title
- Genre
- Language
- Budget
- Popularity
- Release Date
- Revenue
- Runtime
- Vote Count
- Homepage
- Keywords
- Overview
- Production House
- Production Country
- Spoken Language
- Tagline
- Cast
- Crew
- Director

The dataset can be accessed from [this link](https://raw.githubusercontent.com/YBI-Foundation/Dataset/main/Movies%20Recommendation.csv).

## Techniques Used

### Data Preprocessing

- **Feature Selection**: Selecting relevant features such as `Movie_Genre`, `Movie_Keywords`, `Movie_Tagline`, `Movie_Cast`, and `Movie_Director`.
- **Text Vectorization**: Converting text data into numerical format using TF-IDF vectorization.

### Model Building

- **Cosine Similarity**: Calculating the cosine similarity between movies based on their feature vectors to identify similar movies.

## Setup

To set up this project, follow these steps:

1. **Clone the repository**:
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2.pip install pandas numpy scikit-learn
3.import pandas as pd
 import numpy as np
 from sklearn.feature_extraction.text import TfidfVectorizer
 from sklearn.metrics.pairwise import cosine_similarity
4. df = pd.read_csv('path/to/Movies Recommendation.csv')
   df_features = df[['Movie_Genre', 'Movie_Keywords', 'Movie_Tagline', 'Movie_Cast', 'Movie_Director']].fillna('')
   X = df_features['Movie_Genre'] + ' ' + df_features['Movie_Keywords'] + ' ' + df_features['Movie_Tagline'] + ' ' + df_features['Movie_Cast'] + ' ' + df_features['Movie_Director']
   tfidf = TfidfVectorizer()
   X = tfidf.fit_transform(X)
5.df = pd.read_csv('path/to/Movies Recommendation.csv')
  df_features = df[['Movie_Genre', 'Movie_Keywords', 'Movie_Tagline', 'Movie_Cast', 'Movie_Director']].fillna('')
   X = df_features['Movie_Genre'] + ' ' + df_features['Movie_Keywords'] + ' ' + df_features['Movie_Tagline'] + ' ' + df_features['Movie_Cast'] + ' ' + df_features['Movie_Director']
  tfidf = TfidfVectorizer()
  X = tfidf.fit_transform(X)
6.Similarity_Score = cosine_similarity(X)
7.import difflib

Favourie_Movie_Name = input('Enter your favourite movie:')
Movie_Recommendation = difflib.get_close_matches(Favourie_Movie_Name, df['Movie_Title'].tolist())
Close_Match = Movie_Recommendation[0]
Index_of_Close_Match_Movie = df[df.Movie_Title == Close_Match]['Movie_ID'].values[0]
Recommendation_Score = list(enumerate(Similarity_Score[Index_of_Close_Match_Movie]))
Sorted_Similar_Movies = sorted(Recommendation_Score, key=lambda x: x[1], reverse=True)

print('Top 10 Movies suggested for you: \n')
i = 1
for movie in Sorted_Similar_Movies:
    Movie_ID = movie[0]
    Title_From_Index = df[df.Movie_ID == Movie_ID]['Movie_Title'].values
    if i < 11:
        print(i, '.', Title_From_Index)
        i += 1
8.
This README provides a comprehensive overview of the project, including setup and usage instructions. You can customize it further based on your specific needs and the details of your implementation.


