📩 SMS Spam Detection Web App

Live Demo: https://smsspamdetectionn.streamlit.app/

Welcome to the SMS Spam Detection Web App! This application leverages machine learning to classify SMS messages as either Spam or Not Spam. Built using Python, Streamlit, and scikit-learn, it offers an intuitive interface for users to upload datasets and make predictions.

🚀 Features

Model Selection: Choose from various machine learning models, including:

Decision Tree

Random Forest

Multinomial Naive Bayes

Bernoulli Naive Bayes

Automatic Data Processing: Upon uploading a CSV file:

The dataset is loaded and preprocessed.

Features such as word frequencies and message length are extracted.

The data is split into training and testing sets.

Model Training & Evaluation:

Train the selected model on the training data.

Evaluate performance using metrics like accuracy, confusion matrix, and classification report.

Real-time SMS Prediction: Input a new SMS message to predict whether it's spam or not, along with the associated probabilities.

📊 Dataset Format

The application expects a CSV file with the following columns:

label: Indicates whether the message is spam (1) or ham (0).

text: The content of the SMS message.

Example:

label	text
0	Hey, how are you?
1	Congratulations! You've won a prize!
🛠️ Technologies Used

Python: Programming language.

Streamlit: Framework for building the web application.

scikit-learn: Machine learning library.

pandas: Data manipulation and analysis.

📥 How to Use

Upload Dataset: Click on the "Upload your spam dataset CSV" button to upload your CSV file.

Select Model: Choose your preferred machine learning model from the dropdown menu.

Train Model: Click on "Train Model" to train the selected model on your dataset.

Evaluate Model: View the evaluation metrics displayed after training.

Predict SMS: Enter a new SMS message in the text area and click "Predict Spam for Input SMS" to get the prediction.

✅ Conclusion

This SMS Spam Detection Web App provides a simple and interactive way to classify SMS messages using machine learning. It allows users to train models on their own datasets, evaluate performance, and make real-time predictions. By automating feature extraction and providing multiple model choices, the app demonstrates how machine learning can effectively help in filtering unwanted spam messages, making communication safer and more efficient.