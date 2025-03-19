import requests
import base64
import json

url = "http://localhost:8000/v1/chat/completions"

messages = [
    {"role": "user", "text": "Hello! How are you doing today?"},
    {"role": "user", "text": "I'm doing great, thanks for asking! How about you?"},
    {"role": "user", "text": "I'm wonderful, thanks! What are your plans for today?"},
    {"role": "user", "text": "I'm going to work on some coding projects and then relax."}
]

# Assign speakers to each turn
speakers = [4, 2, 1, 2]
max_len = [3000,3000,3000,3000]
payload = {
    "model": "csm-1b",
    "messages": messages,
    "audio": {
        "speakers": speakers,
        "format": "wav",
        "temp": 0.2, # expressiveness
        "min_p": 0.8, # speech clarity
        "max_audio_lens": max_len
    }
}

headers = {"Content-Type": "application/json"}

response = requests.post(url, data=json.dumps(payload), headers=headers)

if response.status_code == 200:
    response_json = response.json()

    # Save each turn's audio
    for i, choice in enumerate(response_json["choices"]):
        filename = f"turn_{i+1}_speaker_{speakers[i]}.wav"
        with open(filename, "wb") as f:
            f.write(base64.b64decode(choice["message"]["audio"]["data"]))
        print(f"Saved: {filename}")

    # Save full conversation audio
    with open("full_conversation.wav", "wb") as f:
        f.write(base64.b64decode(response_json["combined_audio"]))
    print("Saved: full_conversation.wav")

else:
    print("Error:", response.text)
