import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB

st.title("📩 SMS Spam Classifier")

# ----------------------------
# Upload dataset
# ----------------------------
uploaded_file = st.file_uploader("Upload your spam dataset CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file, encoding="latin-1")[['v1','v2']]
    df.columns = ['label','message']

    # Map label to 0/1
    df['is_spam'] = df['label'].map({'ham':0,'spam':1})

    # Extract simple features
    df['word_freq_free'] = df['message'].str.lower().str.count('free')
    df['word_freq_win'] = df['message'].str.lower().str.count('win')
    df['word_freq_offer'] = df['message'].str.lower().str.count('offer')
    df['sms_length'] = df['message'].str.len()

    st.subheader("Sample Data")
    st.dataframe(df.head())

    # Feature/target split
    X = df[['word_freq_free','word_freq_win','word_freq_offer','sms_length']]
    y = df['is_spam']

    # Train/test split (fixed 80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ----------------------------
    # Model selection
    # ----------------------------
    model_choice = st.selectbox("Select Model", ["Decision Tree", "Random Forest", "Multinomial NB", "Bernoulli NB"])

    if model_choice == "Decision Tree":
        model = DecisionTreeClassifier(max_depth=7,min_samples_leaf=2,criterion='entropy',min_samples_split=13)
    elif model_choice == "Random Forest":
        model = RandomForestClassifier(n_estimators=100,max_depth=100)
    elif model_choice == "Multinomial NB":
        model = MultinomialNB(alpha=1.0)
    elif model_choice == "Bernoulli NB":
        model = BernoulliNB(alpha=1.0, binarize=0.0)

    # ----------------------------
    # Train & evaluate
    # ----------------------------
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        st.session_state["trained_model"] = model
        y_pred = model.predict(X_test)

        st.subheader("✅ Model Trained Successfully")
    # ----------------------------
    # Predict new SMS
    # ----------------------------
    st.subheader("🔮 Predict New SMS")
    test_sms = st.text_area("Enter SMS text here:")

    if st.button("Predict Spam for Input SMS"):
        if test_sms.strip() == "":
            st.warning("Please enter a message.")
        elif "trained_model" not in st.session_state:
            st.warning("Please train the model first!")
        else:
            trained_model = st.session_state["trained_model"]
            # Extract features
            new_sms = pd.DataFrame({
                'word_freq_free': [test_sms.lower().count('free')],
                'word_freq_win': [test_sms.lower().count('win')],
                'word_freq_offer': [test_sms.lower().count('offer')],
                'sms_length': [len(test_sms)]
            })
            prediction = trained_model.predict(new_sms)[0]
            prob = trained_model.predict_proba(new_sms) if hasattr(trained_model, "predict_proba") else None

            if prediction == 1:
                st.error("💥 This SMS is predicted as **SPAM**")
            else:
                st.success("💬 This SMS is predicted as **NOT SPAM**")

else:
    st.info("Please upload a spam dataset CSV to start.")
