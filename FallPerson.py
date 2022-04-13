import torch
import cv2
import numpy as np
import os


def load_images_from_folder(folder):
    images = []
    scale_percent = 10 # percent of original size
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder, filename))
        if img is not None:
            width = int(img.shape[1] * scale_percent / 100)
            height = int(img.shape[0] * scale_percent / 100)
            dim = (width, height)
            resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
            images.append(resized)
    return images


def classNameXYWHDetection(results):
    global class_name ,x,y,w,h
    for i, detection in enumerate(results.xywh[0]):
        # detection = tensor (x1,y1,x2,y2,conf,cls)
        x = int(detection[0])
        y = int(detection[1])
        w = int(detection[2])
        h = int(detection[3])
        # numele clasei detectate
        class_name = results.names[int(detection[-1])]
        class_idx = int(detection[-1])
        # print("Class:{}|Class_idx:{}".format(class_name,class_ix))
    return class_name, x, y, w, h


if __name__ == '__main__':
    # cam = cv2.VideoCapture(0)
    imgs_InPicioare = load_images_from_folder(
        "C:/Users/Denis/Desktop/Licenta/Filmari Licenta/InPicioare")
    imgs_cazut = load_images_from_folder(
        "C:/Users/Denis/Desktop/Licenta/Filmari Licenta/Cazut")

  # Incarcarea modelului yolov5 folosind torch.hub
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

   # images = load_images_from_folder("images")
    while True:
        """
        # check, frame = cam.read()

        # results = model(frame)

        for i, detection in enumerate(results.xywh[0]):
            # detection = tensor (x1,y1,x2,y2,conf,cls)
            x = int(detection[0])
            y = int(detection[1])
            w = int(detection[2])
            h = int(detection[3])
            # numele clasei detectate
            class_name = results.names[int(detection[-1])]
            class_idx = int(detection[-1])
            # print("Class:{}|Class_idx:{}".format(class_name,class_ix))

            if class_name == "person":

                    color = (255,140,0)
                    cv2.rectangle(frame, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)),
                                color=color, thickness=2)
                    cv2.putText(frame, class_name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                                color)
            else:
                break

        cv2.imshow("Test", frame)

        key = cv2.waitKey(1)
        if key == 27:
            break
        # Afisarea statisticilor
        # results.print()

    cam.release()
    cv2.destroyAllWindows()"""

        resultInpicioare = model(imgs_InPicioare)
        resultCazut = model(imgs_cazut)
        class_name, x, y, w, h = classNameXYWHDetection(resultInpicioare)
        class_name1, x1, y1, w1, h1 = classNameXYWHDetection(resultCazut)
        target1 = 'C:/Users/Denis/Desktop/Licenta/Filmari Licenta/targetInPicioare'
        target2 = 'C:/Users/Denis/Desktop/Licenta/Filmari Licenta/targetCazut'
        i = 0
        j = 0

        if class_name == "person":
            for img in imgs_InPicioare:
                cv2.imwrite(target1 + "/" + str(
                    i)+'.png', img[int(x-w/2):int(x+w/2), int(y-h/2):int(y+h/2)])
                i += 1
            print(str(i)+' Finalizat')
        if class_name1 == "person":
            for img in imgs_cazut:
                cv2.imwrite(target2 + "/" + str(
                    j)+'.png', img[int(x1-w1/2):int(x1+w1/2), int(y1-h1/2):int(y1+h1/2)])
                j += 1
            print(str(j)+' Finalizat2')
        