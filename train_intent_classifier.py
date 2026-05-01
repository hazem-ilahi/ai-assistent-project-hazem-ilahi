
import random
import nltk
import joblib
from nltk.tokenize import word_tokenize
# from nltk.metrics import ConfusionMatrix
from collections import defaultdict


#  sas-07-intent-classifier.py

# Download the 'punkt' tokenizer model (if you haven't already)
nltk.download('punkt', quiet=True)

def extract_features(text):
    """
    Converts a text string into a Bag-of-Words feature dictionary.
    """
    features = {}
    # Tokenize the text into a list of words
    words = word_tokenize(text.lower())
    
    # Create a feature for the presence of each word
    for word in words:
        features[f'has({word})'] = True
    
    return features

test_text = "is the canteen open now"
print(f"Utterance: '{test_text}'")
print(f"Features:  {extract_features(test_text)}")

# from example dialogs!
training_data = [
    ("hello assistant", "greet"),
    ("hi", "greet"),
    ("hello", "greet"),
    ("assistant", "greet"),
    
    #Use Case: Today's meal list.
    ("what is on menu Today", "meal_today"),
    ("what's for lunch today", "meal_today",),
    ("what are they serving for lunch", "meal_today"),
    ("what food do they have Today", "meal_today"),
    ("today's lunch", "meal_today"),
     
     #vegetarian option and regular menu intents
    ("vegetarian option", "vegetarian"),
    ("regular menu","regular"),
    ("main menu","regular"),
    
    #Use case: Library opening hours.
    ("when does the library opens", "library_hours"),
    ("is the library open right now", "library_hours"),
    ("what time can i go to the library", "library_hours"),
    ("library open time", "library_hours"),
    ("library opening hours please", "library_hours"),
    
    #Use Case: Lecture schedule
    ("when does my lecture start", "lectures_hour"),
    ("lecture schedule please", "lectures_hour"),
    ("is there a lecture today?", "lectures_hour"),
    ("what time is my next lecture", "lectures_hour"),
    ("when do i have a lecture", "lectures_hour"),
     
     #study field
    ("ai","ai_field"),
    ("artificial intelligence", "ai_field"),
    ("cyber security","cy_field"),
    ("cy","cy_field"),

    #Semester
    ("3", "semester_3"),
    ("third semester", "semester_3"),
    ("1", "semester_1"),
    ("first semester", "semester_1"),

    #Use Case: Cafeteria Opening hours
    ("when does the cafeteria close", "cafeteria_hours"),
    ("cafeteria hours", "cafeteria_hours"),
    ("when does the cafeteria opens", "cafeteria_hours"),
    ("when the cafeteria opens", "cafeteria_hours"),
    ("can i still eat at the cafeteria","cafeteria_hours"),

    #Use Case: Language & Study Support
    ("i need help with my studies", "language_study_support"),
    ("is there anything that helps me to develop my language", "language_study_support"),
    ("i need help with learning", "language_study_support"),
    ("i have problems with my studies", "language_study_support"),
    ("i need learning support", "language_study_support"),
     
     #case: study
    ("i need help in my study", "study_support"),
    ("need assistance in my study program","study_support"),
    
    #case: language
    ("i need language courses", "language_support"),
    ("i want to improve my language", "language_support"),
    ("i want to learn language", "language_support"),

    
    
    ("i want to learn english", "english_help"),
    ("i want to learn german", "german_help"),
    ("develop my german skills", "german_help")
]

# Create the feature sets
feature_sets = [
    (extract_features(utterance), intent) 
    for (utterance, intent) in training_data
]

# Shuffle the feature sets for good measure
random.shuffle(feature_sets)

# View the first feature set
print(feature_sets[0])

# Assume 'feature_sets' is your list of (feature_dict, intent) tuples
random.shuffle(feature_sets)

# Split the data (80% training, 20% testing)
split_point = int(len(feature_sets) * 0.8)
train_set = feature_sets[:split_point]
test_set = feature_sets[split_point:]

# Train the Naive Bayes Classifier
classifier = nltk.NaiveBayesClassifier.train(train_set)

accuracy = nltk.classify.util.accuracy(classifier, test_set)
print(f"Overall Accuracy: {accuracy:.2%}")

print("\nMost Informative Features (Mensa Data):")
classifier.show_most_informative_features(5)

# 1. Get predictions for the test set
refsets = defaultdict(set) # Reference (Actual) Intents
testsets = defaultdict(set) # Predicted Intents

for i, (features, intent_actual) in enumerate(test_set):
    intent_predicted = classifier.classify(features)
    refsets[intent_actual].add(i)
    testsets[intent_predicted].add(i)

# 2. Calculate metrics
# NOTE: The list 'intents' should contain all possible intent labels.
intents = classifier.labels()

print(f"intents: {intents}")
print("\n--- Detailed Metrics per Intent ---")
for intent in intents:
    if intent in refsets:
        precision = nltk.scores.precision(refsets[intent], testsets[intent])
        recall = nltk.scores.recall(refsets[intent], testsets[intent])
        f1_score = nltk.scores.f_measure(refsets[intent], testsets[intent])

        # If a metric is None (due to no instances in test set), use 0.0
        print(f"Intent: {intent}")
        print(f"  Precision: {precision if precision else 0.0:.2f}")
        print(f"  Recall:    {recall if recall else 0.0:.2f}")
        print(f"  F1-Score:  {f1_score if f1_score else 0.0:.2f}")

# --- Test our new classifier ---
# test_sentence_1 = "what are the operating hours for the big mensa"
test_sentence_1 = "when the cafeteria opens"
features_1 = extract_features(test_sentence_1)
print(f"'{test_sentence_1}' -> {classifier.classify(features_1)}")

test_sentence_2 = "i need to improve my language skills"
features_2 = extract_features(test_sentence_2)
print(f"'{test_sentence_2}' -> {classifier.classify(features_2)}")

test_sentence_2 = "main menu"
features_2 = extract_features(test_sentence_2)
print(f"'{test_sentence_2}' -> {classifier.classify(features_2)}")

test_sentence_4 = "3"
features_4 = extract_features(test_sentence_4)
print(f"'{test_sentence_4}' -> {classifier.classify(features_4)}")

test_sentence_5 = "1"
features_5 = extract_features(test_sentence_5)
print(f"'{test_sentence_5}' -> {classifier.classify(features_5)}")

test_sentence_6 = "want to study english"
features_6 = extract_features(test_sentence_6)
print(f"'{test_sentence_6}' -> {classifier.classify(features_6)}")

test_sentence_7 = "want to learn german"
features_7 = extract_features(test_sentence_7)
print(f"'{test_sentence_7}' -> {classifier.classify(features_7)}")

test_sentence_8 = "improve my language"
features_8 = extract_features(test_sentence_8)
print(f"'{test_sentence_8}' -> {classifier.classify(features_8)}")



joblib.dump(classifier, "intent_classifier.joblib")
print("Model saved as intent_classifier.joblib")