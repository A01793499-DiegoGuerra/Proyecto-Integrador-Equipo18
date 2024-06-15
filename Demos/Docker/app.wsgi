from flask import Flask, request, redirect

app = Flask(__name__)

@app.route('/')
def index():
    return redirect("http://localhost:8501", code=302)

if __name__ == "__main__":
    app.run()

application = app