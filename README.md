# NLP-with-Flight-Reviews
This is a use case conducted using the data from “4U_Reviews.txt”, which contains 128 review records of customers who took the Germanwings flights between 2008 and 2015.

- The notebook `Overview.ipynb` is used to convert, clean, and visualize the data. It outputs the cleaned review data stored in `Clean_reviews.csv`
- `Classic_sentiment_prediction_approaches.ipynb` uses the `Clean_reviews.csv` to build various text classification models such as TF-IDF, LSTMs, and BERT. It outputs the train and test data used in all models.
**Note**: To run the Bidirectional LSTM model with GloVe embedding, one needs to download the  embedding vector file `glove.6B.50d.txt` from [this website](https://nlp.stanford.edu/projects/glove/).
- In `Topic_modelling_and_sentiment_prediction.ipynb`, the LDA Topic Modelling approach is implemented to classify the different texts into various topics. The output of LDA, i.e., the topic probabilities of each text, is used to train different classifiers. The train and test data sets used for these classifiers are the same as the ones in `Classic_sentiment_prediction_approaches.ipynb`.
**Note**: To open the .html file that contains the interactive visualization for the topic modelling, please use [this link](https://marisshiba.github.io/NLP-with-Flight-Reviews/Topic_visualization.html).
