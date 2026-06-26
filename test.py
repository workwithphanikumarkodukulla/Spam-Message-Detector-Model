import joblib
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load model
model = joblib.load("spam_classifier.pkl")

# Preprocessing
ps = PorterStemmer()
stop_words = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', '', text)
    words = text.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return " ".join(words)

# Test messages
tests = {
    # HAM (0 expected)
    "h1": "Hey, are we still meeting at 6 PM today?",
    "h2": "Please send me the project report when you are free.",
    "h3": "Mom asked you to bring milk on your way home.",
    "h4": "Happy birthday bro! Have a great year ahead.",
    "h5": "The meeting has been moved to tomorrow morning.",
    "h6": "Can you call me when you get time?",
    "h7": "Your Amazon order will be delivered today.",
    "h8": "Don’t forget to submit your assignment.",
    "h9": "Let’s go for dinner tonight.",
    "h10": "I will reach home by 8 PM.",

    # SPAM (1 expected)
    "s1": "URGENT! Your bank account has been suspended. Verify now to restore access.",
    "s2": "Congratulations! You have won a FREE iPhone 16. Click here to claim now.",
    "s3": "Win $5000 cash prize instantly! Reply WIN to receive reward.",
    "s4": "Your PayPal account is locked. Login immediately to avoid termination.",
    "s5": "Limited offer! Buy 1 get 3 free. Hurry up!",
    "s6": "You have been selected for a lottery prize of $10,000. Claim now.",
    "s7": "Earn $1000 per day working from home. No experience needed.",
    "s8": "Your KYC is incomplete. Update now to avoid account suspension.",
    "s9": "Free entry in contest! Text WIN to 12345 now.",
    "s10": "Your SBI account has been blocked. Verify identity immediately.",

    # TRICKY CASES
    "t1": "Hey, congratulations! You won the match yesterday.",
    "t2": "Your meeting is confirmed for tomorrow. Please confirm attendance.",
    "t3": "Congratulations! You have been selected for an exclusive reward program.",
    "t4": "Please verify your email address to continue using the service.",
    "t5": "Urgent: Update your account details to continue access.",
    "t6": "Your parcel will arrive tomorrow. Track it using the link.",
    "t7": "Win exciting prizes by participating in our survey.",
    "t8": "Hey, are you coming to the party tonight?",
    "t9": "Your Netflix subscription will expire soon. Renew to continue service.",
    "t10": "Congratulations on your promotion! Well deserved."
}

print("\n================ MODEL TEST RESULTS ================\n")

for key, msg in tests.items():
    clean = preprocess_text(msg)
    pred = model.predict([clean])[0]

    label = "SPAM 🚫" if pred == 1 else "HAM ✅"

    print(f"{key}: {label}")
    print(f"   Message: {msg}")
    print(f"   Cleaned: {clean}")
    print("-" * 60)