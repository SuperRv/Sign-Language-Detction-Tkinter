from function import *

import cv2
import mediapipe as mp
import tkinter as tk
from keras.models import model_from_json
from keras.models import load_model
from PIL import Image, ImageTk


sequence = []
sentence = []
accuracy=[]
predictions = []
threshold = 0.6 

json_file = open("model.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("model.h5")


root = tk.Tk()
root.title("Sign Language Detection")

# Create a canvas to display the webcam feed
canvas = tk.Canvas(root, width=640, height=480)
canvas.pack()

# Create labels to display detected sign language and accuracy


alphabet_panel = tk.Label(root, text="Sign: ", font=("Helvetica", 24), bg="#68d1e3", padx=20, pady=10)
alphabet_panel.pack(fill=tk.BOTH)

accuracy_label = tk.Label(root, text="Accuracy: ", font=("Helvetica", 14))
accuracy_label.pack(pady=10)

sign_label = tk.Label(root, text="Recent Prediction: ", font=("Helvetica", 14))
sign_label.pack(pady=10)


def update_labels(sign_text, accuracy_text, sentence_text):
    alphabet_panel.config(text="Sign: "+ sign_text)
    accuracy_label.config(text="Accuracy: " + accuracy_text)
    sign_label.config(text="Recent Prediction: " + sentence_text)
    


cap = cv2.VideoCapture(0)

update_labels("Hello","99", "Hello World")


with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    while True:
        ret, frame = cap.read()
        crop_frame = frame[40:400, 0:300]
        frame = cv2.rectangle(frame, (0, 40), (300, 400), (255, 255, 255), 2)

        image, results = mediapipe_detection(crop_frame, hands)

        keypoints = extract_keypoints(results)
        sequence.append(keypoints)
        sequence = sequence[-15:]

        try: 
            if len(sequence) == 15  :
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))
                
                
            #3. Viz logic
                if np.unique(predictions[-1:])[0]==np.argmax(res): 
                    if res[np.argmax(res)] > threshold: 
                        if len(sentence) > 0: 
                            if actions[np.argmax(res)] != sentence[-1]:
                                sentence.append(actions[np.argmax(res)])
                                accuracy.append(str(res[np.argmax(res)]*100))
                                
                        else:
                            sentence.append(actions[np.argmax(res)])
                            accuracy.append(str(res[np.argmax(res)]*100)) 
                            #update_labels(sentence[-1:],accuracy[-1:])

                if len(sentence) > 1: 
                    mainsentence = sentence[-1:]
                    mainaccuracy = accuracy[-1:]

                # Viz probabilities
                # frame = prob_viz(res, actions, frame, colors,threshold)
        except Exception as e:
            # print(e)
            pass

        update_labels( ' '.join(sentence[-1:]),' '.join(accuracy[-1:]), ' '.join(sentence[-5:]))

        # Display the frame on the Tkinter canvas
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (640, 480))
        img = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=img)
        canvas.create_image(0, 0, anchor=tk.NW, image=img)
        canvas.img = img

        # Update the Tkinter window
        root.update_idletasks()
        root.update()

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

root.mainloop()