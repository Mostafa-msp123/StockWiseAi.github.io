import joblib
import os
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Define paths to the model and vectorizer
MODEL_PATH = "chatbot_model.pkl"
VECTORIZER_PATH = "vectorizer.pkl"

class ChatbotModel:
    def __init__(self):
        """Load the model and vectorizer if available, otherwise raise an error."""
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
        else:
            raise FileNotFoundError("❌ Model files not found. Please train the model first.")

    def get_response(self, message):
        """Process the message and return the predicted response."""
        X_input = self.vectorizer.transform([message])
        prediction = self.model.predict(X_input)
        return prediction[0] if prediction else "❌ Sorry, I couldn't find a suitable response."

# Initialize Flask app
app = Flask(__name__, template_folder="templates", static_folder="static")
CORS(app)  # Enable Cross-Origin Requests
chatbot = ChatbotModel()

@app.route("/")
def home():
    return render_template("Index.html")

@app.route("/chat", methods=["POST"])
def chat():
    """API Endpoint for chatbot responses."""
    data = request.json
    user_message = data.get("message", "").strip()

    if not user_message:
        return jsonify({"response": "❌ Please enter a valid question."})

    response = chatbot.get_response(user_message)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
