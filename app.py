# app.py
from flask import Flask, render_template, request
from src.model_setup import rag_chain  # Import the ready-made chain

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/get", methods=["GET", "POST"])
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
