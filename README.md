# Python-Enhanced-Movie-Recommendation-System-

## Introduction
This project aims to develop a basic movie recommendation system using Python and pandas. By leveraging movie ratings and titles datasets, the system suggests movies based on similarity to other films. While not the most advanced recommendation model, it serves as a solid foundation for understanding the mechanics behind content-based recommendation systems.

## Data
The project utilizes two primary datasets:
- A dataset containing user movie ratings.
- A dataset with movie titles and their corresponding ids.

These datasets are merged to form a comprehensive view of user interactions and movie details, enabling the analysis and recommendation process.

## Exploratory Analysis
An initial exploration of the data provides insights into the highest-rated movies and those with the most ratings. This analysis includes:
- Calculating average ratings and the number of ratings per movie.
- Visualizing the distribution of ratings and the number of ratings to understand general user behavior and movie popularity.

Key visualizations are generated using matplotlib and seaborn, offering a glimpse into the typical ratings and the volume of ratings movies receive.

## Recommending Similar Movies
The core of the project lies in creating a matrix that cross-references user ids with movie titles, filled with individual movie ratings. This sparse matrix forms the basis for calculating similarities between movies based on user ratings.

### Methodology
- Utilize the pivot_table function in pandas to create the user-movie interactions matrix.
- Select a couple of movies (e.g., "Star Wars (1977)" and "Liar Liar (1997)") to focus on for generating recommendations.
- Calculate the correlation between movies based on user ratings to find similar movies.
- Clean the resulting correlation data by removing NaN values and filtering movies with a low number of ratings to improve recommendation relevance.

### Results
The system successfully identifies movies similar to "Star Wars (1977)" and "Liar Liar (1997)" based on user ratings, with a logical filtering threshold that ensures recommendations are based on movies with a significant number of ratings. While basic, the recommendations include closely related movies, demonstrating the potential of even simple recommender systems.

## Conclusion
Though this recommender system is elementary, it highlights the potential of using pandas and Python for building content-based recommendation systems. Future iterations could involve more sophisticated algorithms, larger datasets, or even user-based collaborative filtering techniques for improved accuracy.
