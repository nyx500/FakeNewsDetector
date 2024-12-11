# App containing only LIME and natural text explanations for user testing

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

with st.spinner('Downloading required text processing libraries...'):
    progress_bar = st.progress(0)
    nltk.download('wordnet')
    nltk.download('punkt')
    nltk.download('punkt_tab')
    nltk.download('averaged_perceptron_tagger')
    nltk.download('averaged_perceptron_tagger_eng') 
    nltk.download('stopwords')
    # Create set out of English stopwords
stop_words = set(stopwords.words('english'))

# Load the trained model
@st.cache_resource
def load_model():
    return joblib.load('models/my_model.joblib')

# Define a dictionary of regular expression patterns to filter the texts and reduce dimensionality/overly specific patterns
regex_dict = {
    "hashtags": r'#\w+',
    "mentions": r'@[\w-]+',
    "numbers": r'\b\d+\b',
    "emails": r'^[\w\.-]+@([\w-]+\.)+[\w-]{2,4}$',
    "urls": r'https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)',
    "non-http-urls": r'[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&//=]*)',
    "times": r'\b\d{1,2}:\d{2}\s*(?:am|pm|AM|PM)?\b',
    "dates": r'\b(\d{2})/(\d{2})/\d{4}\b',
    "punctuation": r'[^\w\s]',
}

# Helper function to remove matched regexes from samples inside the DataFrame text column
def lowercaseAndFilterText(text):
    """
    Function to apply to 'text' column in pandas DataFrame
    
    Input parameters:
    text (str): text to filter
    
    Output:
    cleaned_text (str): filtered text
    """
    # Ensure 'text' is a string
    text = str(text)
    # Convert text to lowercase
    text = text.lower()
    
    # Apply each regex pattern
    for regex in regex_dict.values():
        text = re.sub(regex, '', text)
    
    # Remove any redundant whitespace
    cleaned_text = re.sub(r'\s+', ' ', text).strip()
    
    return cleaned_text

def preprocessingPipeline(text):
    """
    Preprocess the input text before TF-IDF vectorization.
    
    Input parameters:
    text (str): Text to preprocess
    
    Output:
    processed_text (str): Preprocessed text
    """
    # 1. Convert to lowercase
    # 2. Use regex to remove URL, email, hashtag patterns etc.
    text = lowercaseAndFilterText(text)
    
    # 3. Tokenize with NLTK word_tokenize function
    words = word_tokenize(text)
    
    # 4. Remove stopwords
    no_stopwords = [word for word in words if word.lower() not in stop_words]
    
    # 5. Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in no_stopwords]
    
    return ' '.join(tokens)


# Function for LIME explanation
def generate_lime_explanation(pipeline, text, num_features=30):
    """Modified version of your notebook's LIME function for single text input"""
    explainer = lime.lime_text.LimeTextExplainer(class_names=['real', 'fake'])
    explanation = explainer.explain_instance(text, pipeline.predict_proba, num_features=num_features)

    # Create custom visualization with specific colors
    exp_list = explanation.as_list()


    # Create a DataFrame with features and their corresponding scores
    df = pd.DataFrame(exp_list, columns=['Feature', 'Score'])
    # Sort the DataFrame by the absolute value of scores in descending order
    df['Magnitude'] = df['Score'].apply(lambda x: abs(x))  # Create a column for absolute score
    df_sorted = df.sort_values(by='Magnitude', ascending=False)  # Sort by absolute score in descending order
    # Now we can extract the sorted features and scores for plotting
    features_sorted = df_sorted['Feature'].tolist()
    scores_sorted = df_sorted['Score'].tolist()
    # Plot the sorted bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    # Separate positive (fake) and negative (real) scores into red and blue colors
    colors = ['red' if score > 0 else 'blue' for score in scores_sorted]
    # Create a horizontal bar chart with sorted data
    y_pos = np.arange(len(features_sorted))
    ax.barh(y_pos, scores_sorted, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features_sorted)
    ax.set_xlabel('Word Score (Red = Fake, Blue = Real)')
    ax.set_title('LIME Explanation (Sorted by Importance)')
    plt.gca().invert_yaxis()  # Invert the y-axis to show largest values at top in descending order

    # Return the figure and the list of features + scores
    return fig, exp_list


def convert_to_text(lime_features, prob):
  # Round the probability as a percentage to the nearest integer
  perc = round(prob * 100)
  if prob > 0.5:
    category = "fake"
  elif prob < 0.5:
    category = "real"
  else:
    category = "neither fake nor real"

    
  probability_explanation = f"This news article has been classed as {perc}% fake news, meaning that its binary label would be: '{category.upper()}'. "

  # Words to explain prediction stored in this list
  selected_features= []

  # Iterate over the 50 top LIME features/words to generate explanation
  for feature in lime_features:
    if (feature[1] > 0 and prob > 0.5) or (feature[1] < 0 and prob < 0.5) or (prob == 0.5): # shap_feature[1] contains the SHAP value for this word
        selected_features.append(feature[0]) # Append the word to the selected_features list
    if selected_features == 5:
        break

  # Now we have the selected words, generate the explanation
  feature_explanation = f"""The following words, ranked from high to low in terms of importance, had the most impact in classifying this news text as {category}:
    \n1. {selected_features[0]}\n2. {selected_features[1]}\n3. {selected_features[2]}\n4. {selected_features[3]}\n5. {selected_features[4]}"""

  return probability_explanation + feature_explanation

    
# Set page title
st.title("Explainable Fake News Detection")

# Create tabs
tab1, tab2 = st.tabs(["Analyze Text", "About"])

with tab1:
    # Text input area
    news_text = st.text_area("Enter news text to analyze:", height=200)
    
    # Analysis options
    explanation_type = st.selectbox(
        "Choose explanation type:",
        ["LIME Explainer", "Natural Language"]
    )

     # Analyze button
    if st.button("Analyze Text"):
        if news_text:
            try:
                # Load model and make prediction
                with st.spinner('Loading detection model...'):
                    pipeline = load_model()
                
                with st.spinner('Preprocessing the news text...'):
                    preprocessed_text = preprocessingPipeline(news_text)
                    prediction = pipeline.predict_proba([preprocessed_text])[0]
                
                # Show prediction results
                st.subheader("Results:")
                confidence = prediction[1] * 100
                pred_label = 'Fake' if prediction[1] > 0.5 else 'Real'
                st.markdown(f"**Prediction label:** {pred_label} News")
                st.markdown(f"**Probability that this is fake news**: {confidence:.2f}%")

                with st.spinner("Analyzing important text features and generating explanation..."):
                    figure, features = generate_lime_explanation(pipeline, preprocessed_text)

                # Generate and display LIME explanation
                if explanation_type == "LIME Explainer":
                    with st.spinner('Generating chart...'):
                        st.subheader("LIME Explanation")
                    # Display visualization             
                    st.pyplot(figure)      
                    # Display top features
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

                # Generate and display Natural Language explanation
                else:  # Natural Language
                    st.subheader("Text Explanation")
                    explanation = convert_to_text(features, prediction[1])
                    st.write(explanation)

                                        
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")  

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