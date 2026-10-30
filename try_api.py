import json
import urllib.request
from urllib.parse import unquote

BASE = "http://127.0.0.1:8000"


def post(path, body, content_type):
    request = urllib.request.Request(
        BASE + path, data=body, method="POST", headers={"Content-Type": content_type}
    )
    return urllib.request.urlopen(request)


# 1. Is the server alive?
print("Health:", urllib.request.urlopen(BASE + "/health").read().decode())

# 2. Ask a text question
body = json.dumps({"question": "What credit score does the Home Loan need?"}).encode()
reply = json.loads(post("/ask", body, "application/json").read())
print()
print("Answer:", reply["answer"])
print("Sources:", reply["sources"])

# 3. Send a "voice" question (in this offline mode the audio is just text bytes)
response = post("/voice", b"What is the early closure fee?", "application/octet-stream")
print()
print("Heard: ", unquote(response.headers["X-Transcript"]))
print("Answer:", unquote(response.headers["X-Answer"]))
audio = response.read()
print("Audio: ", len(audio), "bytes,", response.headers["Content-Type"])
