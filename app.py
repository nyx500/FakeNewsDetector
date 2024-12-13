# App containing only LIME and natural text explanations for user testing

# Import required libraries for text preprocessing and Streamlit app building
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import lime
import lime.lime_text
import matplotlib.pyplot as plt
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# st.spinner shows loading spinner for users and the step being executed, to ensure they know something is happening
with st.spinner("Downloading required text processing libraries..."):
    # Download text preprocessing libraries for NLP pipeline
    nltk.download("wordnet")
    nltk.download("punkt")
    nltk.download("punkt_tab")
    nltk.download("averaged_perceptron_tagger")
    nltk.download("averaged_perceptron_tagger_eng") 
    nltk.download("stopwords")
    # Create set out of English stopwords
    stop_words = set(stopwords.words("english"))

# Load the trained model using st.cache_resource
@st.cache_resource
def loadModel():
    return joblib.load("models/my_model2.joblib")

# Define a dictionary of regular expression patterns to filter the texts and reduce dimensionality/overly specific patterns
regex_dict = {
    "hashtags": r"#\w+",
    "mentions": r"@[\w-]+",
    "numbers": r"\b\d+\b",
    "emails": r"^[\w\.-]+@([\w-]+\.)+[\w-]{2,4}$",
    "urls": r"https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)",
    "non-http-urls": r"[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)",
    "times": r"\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b",
    "dates": r"\b(\d{2})/(\d{2})/\d{4}\b",
    "punctuation": r"[^\w\s]",
}

# 
def filterWithRegex(text):
    """
    Lowercases a text and applies regexes to remove unwanted matched patterns from the "text" column in pandas DataFrame
    
    Input Parameters:
        text (str): text to filter
    
    Output:
        cleaned_text (str): filtered text
    """
    # Ensure "text" is a string
    text = str(text)
    # Convert text to lowercase
    text = text.lower()
    
    # Apply each regex pattern
    for regex in regex_dict.values():
        text = re.sub(regex, "", text)
    
    # Remove any redundant whitespace
    cleaned_text = re.sub(r"\s+", " ", text).strip()
    
    return cleaned_text

def preprocessingPipeline(text):
    """
    A function to preprocess the input text using the standard NLP sequence of operations before TF-IDF vectorization.
    
    Input Parameters:
        text (str): Text to preprocess
    
    Output:
        processed_text (str): Preprocessed text
    """

    # 1. Convert to lowercase and 2. Use regex to remove URL, email, hashtag patterns etc.
    text = filterWithRegex(text)
    
    # 3. Tokenize with NLTK word_tokenize function
    words = word_tokenize(text)
    
    # 4. Remove stopwords
    no_stopwords = [word for word in words if word.lower() not in stop_words]
    
    # 5. Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in no_stopwords]
    
    return ' '.join(tokens)


def generateLIMEExplanation(pipeline, text, num_features=30):
    """
        Function for generating LIME scores for feature importance for a single news text and displaying a visualization of the top important features.

        Input Parameters:
            pipeline (scikit-learn Pipeline): a Pipeline containing TF-IDF, PassiveAggressive Classifier and Calibrated Classifier CV steps from the pre-trained model
            text (str): the preprocessed news sample to analyze with LIME explainability
            num_features (int): number of LIME top important features to extract

        Output:
            fig (matplotlib.pyplot plot): visualization to show in the application
            top_features (list): list of tuples containing (word, LIME score)
    """

    # Instantiate the LIME explainer and convert 0 to "real" and 1 to "fake"
    explainer = lime.lime_text.LimeTextExplainer(class_names=['real', 'fake'])
    # Explain the text sample using the LIME explainer object; need to input the preprocessed text, vectorization & model pipeline, and num of top features to extract
    explanation = explainer.explain_instance(text, pipeline.predict_proba, num_features=num_features)

    # Save the LIME explanation as a list of tuples
    top_features = explanation.as_list()

    # Create a DataFrame with the word features and their corresponding LIME scores
    df = pd.DataFrame(top_features, columns=["Word", "Score"])

    # Sort the DataFrame by the absolute value ("Magnitude") of scores in descending order
    df["Magnitude"] = df["Score"].apply(lambda x: abs(x))
    df_sorted = df.sort_values(by="Magnitude", ascending=False)

    # Now we can extract the sorted features and scores for plotting with matplotlib.pyplot
    words_sorted = df_sorted["Word"].tolist()
    scores_sorted = df_sorted["Score"].tolist()

    # Plot the sorted bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    # Separate positive (fake) and negative (real) scores into red and blue colors
    colors = ["red" if score > 0 else "blue" for score in scores_sorted]
    # Create a horizontal bar chart with sorted data
    y_pos = np.arange(len(words_sorted))
    ax.barh(y_pos, scores_sorted, color=colors)
    # Label and annotate the barc chart
    ax.set_yticks(y_pos)
    ax.set_yticklabels(words_sorted)
    ax.set_xlabel('Word Score (Red = Fake, Blue = Real)')
    ax.set_title('LIME Explanation (Sorted by Importance)')
    # Invert the y-axis to show most important words sorted by LIME score at the top
    plt.gca().invert_yaxis()

    # Return the figure and the list of (word, score) tuples
    return fig, top_features


def convertToNaturalLanguageExplanation(lime_features, prob):
  """
    Converts LIME scores to natural language explanations.

    Input Parameters:
        lime_features (list): list of tuples storing (word, LIME score)
        prob (float): probability of news sample being fake

    Output:
        A string containing the natural language explanation.
  """
  # Round the probability as a percentage to the nearest integer
  perc = round(prob * 100)
  if prob > 0.5:
    category = "fake"
  elif prob < 0.5:
    category = "real"
  else:
    category = "neither fake nor real"

  # Write natural language explanation for probability score
  probability_explanation = f"This news article has been classed as {perc}% fake news, meaning that its binary label would be: '{category.upper()}'. "

  # Words to explain prediction stored in this list
  selected_features= []

  # Iterate over the 50 top LIME features/words to generate explanation
  for feature in lime_features:
    # Append words indicating fake news if label is fake, and words pushing towards real news if label is real, or all words if label is neutral (i.e. 0.5)
    if (feature[1] > 0 and prob > 0.5) or (feature[1] < 0 and prob < 0.5) or (prob == 0.5):
        selected_features.append(feature[0])
    # Only get top 5 features
    if selected_features == 5:
        break

  # Now we have the selected words, generate the natural language explanation
  feature_explanation = f"""The following words, ranked from high to low in terms of importance, had the most impact in classifying this news text as {category}:
    \n1. {selected_features[0]}\n2. {selected_features[1]}\n3. {selected_features[2]}\n4. {selected_features[3]}\n5. {selected_features[4]}"""

  return probability_explanation + feature_explanation

    
# Sets the title on the Streamlit application
st.title("Explainable Fake News Detection")

# Create the tabs on the Streamlit application for the analysis page and the application description
tab1, tab2 = st.tabs(["Analyze Text", "About"])

# Do this in the first tab
with tab1:

    # Creates a text input area
    news_text = st.text_area("Enter news text to analyze:", height=200)
    
    # Create a dropdown for analysis options
    explanation_type = st.selectbox(
        "Choose explanation type:",
        ["LIME Explainer", "Natural Language"]
    )

    # This is what happens if the user clicks on the "Analyze Text" button
    if st.button("Analyze Text"):
        if news_text:
            try:
                # Loads the pre-trained model for making the prediction
                with st.spinner('Loading detection model...'):
                    pipeline = loadModel()
                
                # Preprocesses the user's inputted text
                with st.spinner('Preprocessing the news text...'):
                    preprocessed_text = preprocessingPipeline(news_text)
                    prediction = pipeline.predict_proba([preprocessed_text])[0]
                
                # Show prediction results for probability score
                st.subheader("Results:")
                # Converts the probability of being fake news to a percentage
                confidence = prediction[1] * 100
                pred_label = 'Fake' if prediction[1] > 0.5 else 'Real'
                st.markdown(f"**Prediction label:** {pred_label} News")
                st.markdown(f"**Probability that this is fake news**: {confidence:.2f}%")
                
                # Apply the LIME feature importance explanation function to the preprocessed text
                with st.spinner("Analyzing important text features and generating explanation..."):
                    figure, features = generateLIMEExplanation(pipeline, preprocessed_text)

                # Generate and display the LIME explanation
                if explanation_type == "LIME Explainer":
                    # Tell users visualization is being generated
                    with st.spinner('Generating chart...'):
                        st.subheader("LIME Explanation")
                    # Display the visualization using the returned "figure" object
                    st.pyplot(figure)      
                    # Display top features using a bulleted list and the Streamlit Markdown function
                    st.markdown("## Top 10 Ranked Words Influencing Prediction:")
                    markdown_text = ""
                    for feature, score in features[:10]:
                        if score > 0:
                            news_label =  "fake"
                        if score < 0:
                            news_label = "real"
                        if score == 0:
                            news_label = "neutral"
                        markdown_text += f"    - **{feature}**: {score:.3f} -----> (*{news_label} news*)\n"

                    st.markdown(markdown_text)

                # Generate and display the Natural Language explanation if user selects this option instead of LIME visualization
                else:  # Natural Language
                    st.subheader("Text Explanation")
                    explanation = convertToNaturalLanguageExplanation(features, prediction[1])
                    st.write(explanation)
                           
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")  
# In tab2, provide an explanation of the app and some links to how the algorithm works (but this needs to be better explained in layman's terms in the future)
with tab2:
    st.markdown("## About this app:")
    st.markdown(" - This app uses machine learning to detect fake news and explain its decisions.")
    st.markdown(" - This model has been trained on the WELFake dataset and explanations have been generated using LIME "
                "(Local Interpretable Model-agnostic Explanations).")
    st.markdown(" - While treating the machine-learning model as a black box, LIME perturbs (slightly changes) the text we "
                "want to explain and learns the impact of changing each feature (word) on the prediction.")
    st.markdown(" - You can find the (WELFake) dataset used for training [here](https://zenodo.org/records/4561253)")
    st.markdown(" - Please check out [this paper](https://arxiv.org/pdf/1602.04938) for a more detailed explanation of how "
                "LIME works.")