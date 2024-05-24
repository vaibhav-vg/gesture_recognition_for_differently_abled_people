from tkinter import *
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image, ImageTk 
import pickle

model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

root = Tk()
root.title("Hand Sign Prediction")


# Title.pack()
vid = cv2.VideoCapture(0) 
  
width, height = 800, 600
  
vid.set(cv2.CAP_PROP_FRAME_WIDTH, width) 
vid.set(cv2.CAP_PROP_FRAME_HEIGHT, height) 
  
bg = PhotoImage(file = "proj_img_4k.png", master=root) 
  
label1 = Label( root, image = bg) 
label1.place(x=0,y=0)
Title = Label(root, text = "Gesture Recognition System", font=("Helvetica",36,"bold"))

Title.pack()
  
root.bind('<Escape>', lambda e: root.quit()) 
  
label_widget = Label(root) 
label_widget.pack() 


def center_text(text_widget):
    text = text_widget.get("1.0", END).strip()  # Get and trim text
    if not text:  # Handle empty text case
        return

    width = text_widget.winfo_width()  # Get widget width
    text_width = text_widget.measure("current")  # Measure text width

    # Calculate padding for centered alignment
    padding = (width - text_width) // 2
    text_widget.tag_configure("center", justify=Tk.CENTER, spacing=padding)
    text_widget.insert("1.0", text, "center")

def prediction(image):
    labels_dict = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}
    data_aux = []
    x_ = []
    y_ = []
    ret, frame = True, image
    # H, W, _ = 0, 0, 0
    if ret:
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

            expected_features = 42
            data_aux = np.asarray(data_aux)
            if data_aux.shape[0] > expected_features:
                data_aux = data_aux[:expected_features]

            prediction = model.predict([np.asarray(data_aux)])

            predicted_character = labels_dict[int(prediction[0])]
            # print(predicted_character)
            T.delete(0.0,END)
            T.insert(INSERT,predicted_character)
            T.bind("<KeyRelease>", lambda event: center_text(T))
            # T.anchor(CENTER)
    return 

  
def open_camera(): 
    _, frame = vid.read() 
    
    prediction(frame)
    
    opencv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA) 
  
    captured_image = Image.fromarray(opencv_image) 
  
    photo_image = ImageTk.PhotoImage(image=captured_image) 
  
    label_widget.photo_image = photo_image 
  
    label_widget.configure(image=photo_image) 
  
    label_widget.after(10, open_camera) 
    
    label_widget.anchor
  
button1 = Button(root, text="Open Camera", relief='groove', font=("Arial", 12, ), command=open_camera)
button1.pack(pady=10)


T = Text(root, height = 1, width = 2)
T.config(font=("Helvetica",24,"bold"))
 
# Create label
l = Label(root, text = "Predicted character",padx="1",pady=2)
l.config(font =("Courier", 14))

l.pack()
T.pack()


root.mainloop()