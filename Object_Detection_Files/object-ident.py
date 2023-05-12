# Import the Open-CV extra functionalities
import cv2
# import RPi.GPIO as GPIO
# Define the LED GPIO pin
# LED_PIN = 18

# Set up the GPIO
# GPIO.setmode(GPIO.BCM)
# GPIO.setup(LED_PIN, GPIO.OUT)
# This is to pull the information about what each object is called
classNames = []
classFile = "/home/pi/Desktop/Object_Detection_Files/coco.names"

with open(classFile, "rt") as f:
    classNames = f.read().rstrip("\n").split("\n")

# This is to pull the information about what each object should look like
# configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
configPath = "/home/pi/Desktop/Object_Detection_Files/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
weightsPath = "/home/pi/Desktop/Object_Detection_Files/frozen_inference_graph.pb"

# This is some set up values to get good results
net = cv2.dnn_DetectionModel(weightsPath, configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)


def getObjects(img, thres, nms, draw=True, objects=[], confidence_threshold=0.6):
    classIds, confs, bbox = net.detect(img, confThreshold=thres, nmsThreshold=nms)
    # Below has been commented out, if you want to print each sighting of an object to the console you can uncomment below
    # print(classIds,bbox)
    if len(objects) == 0: objects = classNames
    objectInfo = []
    personDetected = False
    if len(classIds) != 0:
        for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
            className = classNames[classId - 1]
            if className in objects and confidence >= confidence_threshold:
                personDetected = True
                objectInfo.append([box, className])
                if (draw):
                    cv2.rectangle(img, box, color=(0, 255, 0), thickness=2)
                    label = f"{classNames[classId - 1].upper()} {int(confidence * 100)}%"
                    cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if personDetected:
            print("person is detected")
            # GPIO.output(LED_PIN, GPIO.HIGH)
        else:
            print("person is not detected")
            # GPIO.output(LED_PIN, GPIO.LOW)

    return img, objectInfo


# Below determines the size of the live feed window that will be displayed on the Raspberry Pi OS
if __name__ == "__main__":
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)
    # cap.set(10,70)

    # Below is the never ending loop that determines what will happen when an object is identified.
    while True:
        success, img = cap.read()
        # Below provides a huge amount of controll. the 0.45 number is the threshold number, the 0.2 number is the nms number)
        result, objectInfo = getObjects(img, 0.45, 0.2, objects=['person'])
        # print(objectInfo)
        cv2.imshow("Output", img)
        k = cv2.waitKey(1)
        if k == 27 or cv2.getWindowProperty("Output", cv2.WND_PROP_VISIBLE) < 1:  # wait for ESC key or window close button to exit
            break

    cap.release()

