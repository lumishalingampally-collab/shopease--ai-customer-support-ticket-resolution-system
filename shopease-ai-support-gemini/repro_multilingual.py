
import requests

BASE_URL = "http://127.0.0.1:8000"

def test_language(query, expected_lang_name):
    print(f"\nTesting Query: {query}")
    response = requests.post(f"{BASE_URL}/api/v1/chat", json={"query": query})
    if response.status_code == 200:
        data = response.json()
        print(f"Detected Language: {data['classification']['language']}")
        print(f"Response: {data['response']}")
        # Simple check for script
        if expected_lang_name == "Hindi":
            # Check for Devanagari characters
            has_devanagari = any('\u0900' <= c <= '\u097F' for c in data['response'])
            print(f"Has Hindi script: {has_devanagari}")
        elif expected_lang_name == "Telugu":
            # Check for Telugu characters
            has_telugu = any('\u0C00' <= c <= '\u0C7F' for c in data['response'])
            print(f"Has Telugu script: {has_telugu}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # Hindi query
    test_language("मुझे अपना सामान वापस करना है क्या नियम है?", "Hindi")
    
    # Telugu query
    test_language("నా ఆర్డర్ ఇంకా రాలేదు డెలివరీ ఎప్పుడు వస్తుంది?", "Telugu")
