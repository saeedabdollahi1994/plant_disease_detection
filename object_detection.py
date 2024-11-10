from ultralytics import YOLO


class MyClass:
    def __init__(self):
        self.output = []

    def object_detection(self,image):
        model = YOLO("best.pt")
        results = model(image)
        names = model.names
        for r in results:
            for c in r.boxes.cls:
                self.output.append(names[int(c)])
                

        return self.output,results[0].plot()