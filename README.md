# MoodLens — Sentiment & Emotion Detection

MoodLens is a Streamlit-based application that predicts the emotion expressed in a piece of text. It uses NLP techniques and a Logistic Regression classifier to analyze text and display the predicted emotion along with probability charts and model evaluation results.

## Live Demo

Try MoodLens here: [**Launch MoodLens**](https://sentiment-detection-nlp-94ngdnkki89cnmgykapj6n.streamlit.app/)

Explore the app by entering text and seeing which emotion the model predicts.

## Features

- **Emotion prediction:** Enter a sentence and get the model's predicted emotion.
- **Prediction probabilities:** View the model's confidence scores across emotion classes.
- **Dataset insights:** Explore the labelled training data.
- **Model evaluation:** Check the classifier's performance on test data.
- **Interactive interface:** Use the application directly through a Streamlit web interface.

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- TF-IDF Vectorization
- Logistic Regression
- Pandas, NumPy and visualization libraries, where used

## Project Structure

```text
sentiment_detection_streamlit/
├── app.py
├── train.txt
├── requirements.txt
├── README.md
└── .gitignore
```

## Getting Started

### Prerequisites

Install Python 3.10 or newer.

### Installation

1. Clone this repository:

   ```bash
   git clone <your-repository-url>
   ```

2. Move into the project directory:

   ```bash
   cd sentiment_detection_streamlit
   ```

3. (Optional) Create a virtual environment:

   ```bash
   python -m venv venv
   ```

4. Activate the environment.

   **Windows:**
   ```bash
   venv\Scripts\activate
   ```

5. Install the required libraries:

   ```bash
   pip install -r requirements.txt
   ```

### Run the Application

Start the Streamlit app using:

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually `http://localhost:8501`.

## How It Works

1. The application loads the labelled data from `train.txt`.
2. Text is prepared for model training while preserving important words such as negations.
3. TF-IDF converts the text into numerical features.
4. Logistic Regression learns to classify the text into emotion categories.
5. The model is evaluated using a stratified 80/20 train-test split.
6. After evaluation, the classifier is retrained on the complete dataset for interactive predictions.

The dataset should contain one example per row in the format `text;emotion`.

## Model Evaluation

The application reports test accuracy based on the held-out test split. Add the actual accuracy and any other evaluation metrics here after running the application.

## Limitations

The predicted emotion is an estimate based on patterns learned from the training data. Sarcasm, slang, ambiguous wording and sentences expressing multiple emotions may lead to incorrect predictions.

## Future Improvements

- Test the model on more diverse text samples.
- Compare Logistic Regression with other classification algorithms.
- Improve performance on ambiguous and informal language.
- Deploy the application for public access.

## Author

Bishakha Prasad

