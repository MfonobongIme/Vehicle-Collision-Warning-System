#importing the libraries we need
import cv2
import numpy as np
import numpy as np
from PIL import ImageGrab
import cv2
import time
import random

net = cv2.dnn.readNet('yolov3.weights', 'yolov3.cfg')

#extract the object names from the coco files into a list
classes = []
with open('coco.names', 'r') as f:
    classes = f.read().splitlines()


#img = cv2.imread('office.jpg')
img_width = 800 #image width
img_height = 668 #image width

roi_width = 800 #region of interest width
roi_height = 420 #region of interest height

#generate a random message list for Jarvis
message = ['Jarvis: Collision Alert!!']
messageRan = random.choice(message)

#grab screen function
while True:
    img = printscreen_pil = np.array(ImageGrab.grab(bbox=(0, 100, 800, 668)))
    # printscreen_numpy = np.array(printscreen_pil.getdata(), dtype = 'uint8')
    img_roi = np.copy(img)
    img_roi = img_roi[0:420, :, :]
    #print(img_roi)

    last_time = time.time()
    #print('Loop took {} seconds'.format(time.time() - last_time))


    height, width, _ = img_roi.shape #extract the width and height of image

    blob = cv2.dnn.blobFromImage(img_roi, 1/255, (224, 224), (0, 0, 0), swapRB=True, crop=False)

    #pass the blob immage into the network
    net.setInput(blob) #sets input from the blob into the network
    output_layers_names = net.getUnconnectedOutLayersNames() #get the output layers names
    layerOutputs = net.forward(output_layers_names) #runs the forward pass and obtain the output at the output layer which we already provided the layers names

    boxes = [] #box list to extract the bounding boxes
    confidences = [] #confidence list to store the confidence

    class_ids = [] #class list which stores the predicted classes



    for output in layerOutputs: #used to extract all info from the layeroutput
        for detection in output: #used to extract the information in each of the outputs
            scores = detection[5:] #store all the 80 classes predictions starting from the 6th element till the end
            class_id = np.argmax(scores)#identify the classes that has the highest scores in scores list
            confidence = scores[class_id] #pass these elements to identify the maximum value from these scores which is the probability
            if confidence > 0.: #set confidence threshold
                center_x = int(detection[0]*width) #multiply by width and height to rescale it back
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)

                #since YOLO gets positions of objects from its center and opencv gets positions of objects
                #from the upper left, we need to perform this calculations for opencv to get its positions
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                #append all the information to the corresponding list
                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)


    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4) #gets rid of redundant boxes
    if len(indexes) > 0:
        #print(indexes.flatten())
        print(len(indexes))
        #print('good')

    font = cv2.FONT_HERSHEY_SIMPLEX
    colors  = np.random.uniform(0, 255, size=(len(boxes), 3))

    #create a for loop to loop over all the objects detected

    if len(indexes) > 0:

        for i in indexes.flatten():

            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i], 2))
            color = colors[i]




            print(boxes[i])


            #print(class_ids)
            if class_ids[i] == 2 or class_ids[i] == 5 or class_ids[i] == 9 or class_ids[i] == 0 or class_ids[i] == 3:
                mid_x = (w+x)/2
                mid_y = (h+y)/2

                print(mid_x)
                print(mid_y)

                apx_distance = (y-h)
                print(apx_distance)

                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
                cv2.putText(img, label, (x, y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                if apx_distance <= 210:
                    if mid_x > 200 and mid_x < 300:
                        #cv2.rectangle(img, (300, 0), (600, 60), (245, 117, 16), -1)
                        ##           cv2.LINE_AA)


                        cv2.putText(img, 'WARNING!!', (int(mid_x+100), int(mid_y+170)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,230,0), 2)
                        cv2.putText(img, str(apx_distance / 100) + 'metres', (int(mid_x + 90), int(mid_y + 200)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

                        cv2.putText(img, 'Jarvis: {}'.format(str(apx_distance / 100) + 'meters to impact'), (270, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                                    2,
                                    cv2.LINE_AA)
                        cv2.putText(img, messageRan, (270, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                    (255, 255, 255), 2,
                                    cv2.LINE_AA)
                        cv2.line(img, (260, 0), (260, 70), color, 2)
                        cv2.line(img, (600, 0), (600, 70), color, 2)









                    if apx_distance <= 195:
                        if mid_x > 200 and mid_x < 300:
                            cv2.putText(img, 'WARNING!!', (int(mid_x + 100), int(mid_y + 170)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                            cv2.putText(img, str(apx_distance/100) + 'meters', (int(mid_x + 90), int(mid_y + 200)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
                            cv2.putText(img, str(messageRan), (270, 22),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                        (255, 255, 255), 2,
                                        cv2.LINE_AA)

                            cv2.putText(img, 'Jarvis: {}'.format(str(apx_distance / 100) + 'meters to impact'), (270, 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255),
                                        2,
                                        cv2.LINE_AA)





        cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        #cv2.imshow('image_roi', cv2.cvtColor(img_roi, cv2.COLOR_BGR2RGB))

        key = cv2.waitKey(1)
        if key == 27:
            break
cap.release()
cv2.destroyAllWindows()