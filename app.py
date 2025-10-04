from flask import Flask, render_template, request
from model_setup import rag_chain  # Import the retrieval-augmented generation chain
import os

app = Flask(__name__)

# ------------------------
# Routes
# ------------------------
@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["POST"])
def chat():
    msg = request.form["msg"]
    print("User:", msg)

    try:
        response = rag_chain.invoke({"input": msg})
        print("Response:", response["answer"])
        return str(response["answer"])
    except Exception as e:
        print("Error:", e)
        return "Sorry, the server encountered an error."


# ------------------------
# Run app
# ------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Use Render's PORT if available
    app.run(host="0.0.0.0", port=port)
