import tkinter as tk
from tkinter import filedialog
from PIL import ImageTk, Image
import cv2
import numpy as np


CONFIDENCE = 0.5
IOU_THRESHOLD = 0.5
SCORE_THRESHOLD = 0.5
config_path = "yolov3-helmet.cfg"
weights = "yolov3-helmet.weights"
labels = open("helmet.names").read().strip().split("\n")

net = cv2.dnn.readNetFromDarknet(config_path, weights)


def model_output(path_name):
    image = cv2.imread(path_name)
    h,w = image.shape[:2]
    
    
    blob = cv2.dnn.blobFromImage(image, 1/255.0,(416,416), swapRB = True, crop = False)
    
    net.setInput(blob)  # Sets the new input value for the network
    
    ln = net.getLayerNames()
    ln = [ln[i-1] for i in list(net.getUnconnectedOutLayers())]
    layer_outputs = net.forward(ln)
    boxes, confidences, class_ids = [], [], []
    for output in layer_outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence>CONFIDENCE:
                box = detection[:4]*np.array([w,h,w,h])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width/2))
                y = int(centerY - (height/2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    return boxes, confidences, class_ids


def detection_recognition(path_name):
    image = cv2.imread(path_name)
    boxes, confidences, class_ids = model_output(path_name)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, SCORE_THRESHOLD, IOU_THRESHOLD)
    font_scale = 1
    thickness= 1
    if len(idxs)>0:
        for i in idxs.flatten():
            x, y, w, h = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
            cv2.rectangle(image, (x,y), (x+w, y+h), color = (255,20,147), thickness = thickness)
            text = f"{labels[class_ids[i]]}:{confidences[i]:.2f}"
            (text_width, text_height) = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, fontScale = font_scale, thickness = thickness)[0]
            text_offset_x = x
            text_offset_y = y - 5
            box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height))
            overlay = image.copy()
            cv2.rectangle(overlay, box_coords[0], box_coords[1], color = (255,20,147), thickness = cv2.FILLED)
            image = cv2.addWeighted(overlay, 0.6, image, 0.4, 0)
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=font_scale, color=(0, 0, 0), thickness=thickness)
    
    # Resize the image to fit within a 400x400 pixel window
    h, w = image.shape[:2]
    if h > w:
        new_h = 400
        new_w = int(w * (new_h / h))
    else:
        new_w = 400
        new_h = int(h * (new_w / w))
    image = cv2.resize(image, (new_w, new_h))
    
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)



# create the main window
root = tk.Tk()
root.title("Helmet Detection")
root.geometry("800x700")

# create a function to open the file dialog and load the image
def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        detection_image = detection_recognition(file_path)
        photo = ImageTk.PhotoImage(Image.fromarray(detection_image))
        label.config(image=photo)
        label.image = photo



# create a button to open the camera and capture an image

def open_camera():
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    capture = False # set capture flag to False
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow('Camera', frame)
        key = cv2.waitKey(1)
        if key == ord(' '):
            capture = True # set capture flag to True
        if capture: # if capture flag is True
            # save the captured image to a temporary file
            temp_file = 'temp.jpg'
            cv2.imwrite(temp_file, frame)
            cap.release()
            cv2.destroyAllWindows()
            detection_image = detection_recognition(temp_file)
            photo = ImageTk.PhotoImage(Image.fromarray(detection_image))
            label.config(image=photo)
            label.image = photo
            break
        elif key == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            break


# create a button to toggle the background color of the window
def toggle_background():
    current_color = root.cget("bg")
    if current_color == "#ffffff":
        root.config(bg="#000000")
        open_button.config(bg="#1e1e1e", fg="#ffffff", activebackground="#ffffff", activeforeground="#000000")
        camera_button.config(bg="#000000", fg="#ffffff", activebackground="#ffffff", activeforeground="#000000")
        label.config(bg="#000000")
        header.config(bg="#000000", fg="#ffffff")
        footer.config(bg="#000000", fg="#808080")
        button_frame.config(bg="#1e1e1e")
    else:
        root.config(bg="#ffffff")
        open_button.config(bg="#ffffff", fg="#000000", activebackground="#000000", activeforeground="#ffffff")
        camera_button.config(bg="#ffffff", fg="#000000", activebackground="#000000", activeforeground="#ffffff")
        label.config(bg="#ffffff")
        header.config(bg="#ffffff", fg="#000000")
        footer.config(bg="#ffffff", fg="#808080")
        button_frame.config(bg="lightgrey")


# create a label to display the image


header = tk.Label(root, text="Helmet Detection", font=("Segoe UI", 26), bg="#1e1e1e", fg="white", pady=20)
header.pack(fill="x")


#add a frame to contain the buttons
button_frame = tk.Frame(root, bg="#1e1e1e", pady=10)
button_frame.pack(fill="x")

#create a button to open the file dialog and load the image
open_button = tk.Button(button_frame, text="Choose Image", command=open_image, font=("Segoe UI", 14), padx=10, pady=5, bg="#3b3b3b", fg="white", bd=0, activebackground="#565656", activeforeground="white")
open_button.pack(side="left", padx=10)

#create a button to open the camera and capture an image
camera_button = tk.Button(button_frame, text="Open Camera", command=open_camera, font=("Segoe UI", 14), padx=10, pady=5, bg="#3b3b3b", fg="white", bd=0, activebackground="#565656", activeforeground="white")
camera_button.pack(side="left", padx=10)

#create a button to toggle the background color of the window
background_button = tk.Button(button_frame, text="Change Theme", command=toggle_background, font=("Segoe UI", 14), padx=10, pady=5, bg="#3b3b3b", fg="white", bd=0, activebackground="#565656", activeforeground="white")
background_button.pack(side="right", padx=10)

#add a label for the image
label = tk.Label(root, bg="#2d2d2d", padx=10, pady=10)
label.pack(expand=True, fill="both")

#add a footer label
footer = tk.Label(root, text="Made by Nimit Banka", font=("Segoe UI", 10), bg="#1e1e1e", fg="#707070", pady=10)
footer.pack(fill="x", side="bottom")

#set the background color of the window
root.config(bg="#2d2d2d")

#start the main loop
root.mainloop()