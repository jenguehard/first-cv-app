import streamlit as st 
import cv2
import numpy as np
import time
import os

labelsPath = "classes.names"
LABELS = open(labelsPath).read().strip().split("\n")
colors = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

weightsPath = "yolov4-tiny.weights"
configPath = "yolov4-tiny.cfg"
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)


##### FUNCTION #####

def get_classification(img, net, LABELS, conf_threshold) :
    start = time.time()
    image = img
    (H, W) = image.shape[:2]
    if W < 1000:
        font_size = 1
        thickness = 1
    elif W > 1000 and W < 2000:
        font_size = 2
        thickness = 2
    else :
        font_size = 5
        thickness = 5 
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
	swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)
    end = time.time()
	
    boxes = []
    confidences = []
    classIDs = []
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            if confidence > conf_threshold :
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold,
	0.3)
    if len(idxs) > 0:
        for i in idxs.flatten():
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])

            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
            cv2.putText(
                image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_size, color, 2
            )
            label = "Inference Time: {:.2f} s".format(end - start)
            # cv2.putText(
            #     image, label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2
            # )
    return image

#####
st.title("Use Yolov4 to detect object on an image")

st.write("Yolov4 is trained on COCO dataset and is able to detect 80 classes of elements (animals, objects...). Play with the confidence of the algorithm to check if you get the same predictions !")
st.write("")
st.write("If you choose a high value, you will get less predictions but more accurate. On the other hand, if you choose a low value, you will get more predictions that could be less accurate.")
st.write("")

conf_threshold = st.slider('Choose the confidence of the algorithm.', 0.0, 1.0, 0.5, 0.05)

st.header('Choose your image')


uploaded_file = st.file_uploader("Choose an image...", type=['png','jpeg','jpg'])

pictures_list = [i.split(".jpg")[0] for i in os.listdir("Pictures")]

if uploaded_file is None:
    option = st.selectbox('...or choose one of mine !', pictures_list)
    img_path = os.path.join("Pictures", option + ".jpg")
    img = cv2.imread(img_path)
    st.image(img, channels="BGR", use_column_width=True)

    if st.button("Predict !"):
        predicted_image = get_classification(img, net, LABELS, conf_threshold)
        st.image(predicted_image, channels="BGR", use_column_width=True)

else:
    # Convert the file to an opencv image.
    filename = uploaded_file.name
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)

    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="BGR", use_column_width=True)
    # cv2.imwrite(filename, opencv_image)

    if st.button("Predict !"):
        predicted_image = get_classification(opencv_image, net, LABELS, conf_threshold)
        st.image(predicted_image, channels="BGR", use_column_width=True)
