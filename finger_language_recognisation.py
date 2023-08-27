import pickle

import cv2
import mediapipe as mp
import numpy as np
import openai
from googleapiclient.discovery import build

my_api_key = "AIzaSyBc35KWebG-mhquGQeZML-_J3g5VyDurQA" #The API_KEY
my_cse_id = "305af84e382b84402" #The search-engine-ID
openai.api_key = "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"

def google_search(search_term, api_key, cse_id, **kwargs):
    service = build("customsearch", "v1", developerKey=api_key)
    res = service.cse().list(q=search_term, cx=cse_id, **kwargs).execute()
    return res['items']

def chat_with_chatgpt(prompt, model="gpt-3.5-turbo"):
    response = openai.Completion.create(
        engine=model,
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )
    message = response.choices[0].text.strip()
    return message

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

cap = cv2.VideoCapture(2)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)
#labels_dict = {0: ',', 1: ' ', 2: 'A', 3: 'C', 4: 'I', 5: 'L', 6: 'O', 7: 'P', 8: 'T', 9: 'U', 10: 'S'}
#character_counts = {',': 0, ' ': 0, 'A': 0, 'C': 0, 'I': 0, 'L': 0, 'O': 0, 'P': 0, 'T': 0, 'U': 0, 'S': 0}
labels_dict = {0: ',', 1: ' ', 2: 'n', 3: 'e', 4: 'w', 5: 's'}
character_counts = {',': 0, ' ': 0, 'n': 0, 'e': 0, 'w': 0, 's': 0}
queryQ = ""
numLetter = 0
repeat = 50

while True:

    data_aux = []
    x_ = []
    y_ = []

    ret, frame = cap.read()

    H, W, _ = frame.shape

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = hands.process(frame_rgb)
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,  # image to draw
                hand_landmarks,  # model output
                mp_hands.HAND_CONNECTIONS,  # hand connections
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

        for hand_landmarks in results.multi_hand_landmarks:
            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y

                x_.append(x)
                y_.append(y)

            for i in range(len(hand_landmarks.landmark)):
                x = hand_landmarks.landmark[i].x
                y = hand_landmarks.landmark[i].y
                data_aux.append(x - min(x_))
                data_aux.append(y - min(y_))

        x1 = int(min(x_) * W) - 10
        y1 = int(min(y_) * H) - 10

        x2 = int(max(x_) * W) - 10
        y2 = int(max(y_) * H) - 10

        prediction = model.predict([np.asarray(data_aux)])
        
        predicted_character = labels_dict[int(prediction[0])]

        if predicted_character in character_counts:
            character_counts[predicted_character] += 1

        for char, count in character_counts.items():
            if count > repeat:
                if numLetter == 0:
                    print("\n===============================")
                    print("User: ", end = "", flush=True)
                    print(char.capitalize(), end = "", flush=True)
                else:
                    print(char, end = "", flush=True)
                queryQ += char
                character_counts = dict.fromkeys(character_counts, 0)
                numLetter += 1

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
        cv2.putText(frame, predicted_character, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 0, 0), 3,
                    cv2.LINE_AA)
    else:
        if numLetter > 40:
            print(".")
            numLetter = 0


    cv2.imshow('frame', frame)
    cv2.waitKey(10)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

    if cv2.waitKey(10) & 0xFF == ord('g'):
        retries = 4
        while retries > 0 : 
            try : 
                results = google_search(queryQ, my_api_key, my_cse_id, num=10) # "top news" "upcoming events"
                print(".")
                print(f"Google: {results[0]['title']}")
                print("\n===============================")
                break
            except:
              retries = retries - 1

        queryQ = ""
        numLetter = 0

    if cv2.waitKey(10) & 0xFF == ord('o'):
        messages = [ {"role": "system", "content": "You are a intelligent assistant."} ]
        # while True:
        message = queryQ # input("User : ")
        if message:
            messages.append(
                {"role": "user", "content": message},
            )
            chat = openai.ChatCompletion.create(
                model="gpt-3.5-turbo", messages=messages
            )

        reply = chat.choices[0].message.content
        print(f"ChatGPT: {reply}")
        print("===============================")
        messages.append({"role": "assistant", "content": reply})

        list(filter(lambda x: x["role"] == "assistant",messages))[0]["content"]

        queryQ = ""
        numLetter = 0


cap.release()
cv2.destroyAllWindows()
