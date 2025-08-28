# very small HTTP function (Flask style for faas)
from flask import Flask, request, jsonify
app = Flask(__name__)

@app.route('/', methods=['POST','GET'])
def handle():
    # simulate work (0.1s - 1s)
    import random, time
    t = random.uniform(0.1, 1.0)
    time.sleep(t)
    return jsonify({"ok": True, "work": t})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
