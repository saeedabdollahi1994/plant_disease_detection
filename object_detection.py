from ultralytics import YOLO
import cv2 as cv

class MyClass:
    def __init__(self):
        self.output = []

    def object_detection(self,image):
        model = YOLO("best.pt")
       #image = cv.imread(image)
        results = model(image)
        names = model.names
        for r in results:
            for c in r.boxes.cls:
                self.output.append(c)
                #print(int(c),names[int(c)])

        return self.output,results[0].plot()