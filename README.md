# Spam Email Detection using ML

A simple machine learning web application that classifies emails as **Spam** or **Not Spam** using **TF-IDF Vectorization** and the **Multinomial Naive Bayes** algorithm. The application is built with Flask and provides predictions through a simple web interface.

## Features

- Classifies emails as Spam or Not Spam
- Displays prediction confidence
- User-friendly web interface
- Built using Flask and Scikit-learn

## Tech Stack

- Python
- Flask
- Scikit-learn
- Pandas
- HTML
- CSS

## Project Structure

```
Spam_Email_Classifier/
│
├── dataset/
│   └── spam.tsv
│
├── static/
│   ├── style.css
│   └── script.js
│
├── templates/
│   └── index.html
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
└── README.md
```
## Dataset

The project uses the SMS Spam Collection Dataset, which contains labeled SMS messages categorized as **Spam** or **Ham (Not Spam)**.


## Workflow

**Training**
Dataset → Preprocessing → TF-IDF → Naive Bayes → Saved Model

**Prediction**
User Input → TF-IDF → Trained Model → Spam / Not Spam

## How to Run

1. Clone the repository.

```bash
git clone https://github.com/tejalakshmi6/Spam-Email-Classifier.git
```

2. Move to the project folder.

```bash
cd Spam-Email-Classifier
```

3. Install the required libraries.

```bash
pip install -r requirements.txt
```

4. Train the model.

```bash
python train_model.py
```

5. Run the Flask application.

```bash
python app.py
```

6. Open your browser and go to:

```
http://127.0.0.1:5000
```

## Model

- **Algorithm:** Multinomial Naive Bayes
- **Feature Extraction:** TF-IDF Vectorizer
- **Accuracy:** 96.68%

## Screenshots

### Home Page

<img width="1364" height="619" alt="image" src="https://github.com/user-attachments/assets/85a3359c-3b88-46c6-9444-738f62875bc0" />


### Prediction Result

<img width="1365" height="628" alt="image" src="https://github.com/user-attachments/assets/e6e447f0-e211-4ffb-a51f-36368f4909d2" />

<img width="1365" height="629" alt="image" src="https://github.com/user-attachments/assets/5f6cec7b-10c1-4ff2-bcc3-c97d0da38901" />

## Author

**Tejalakshmi**

GitHub: https://github.com/tejalakshmi6
