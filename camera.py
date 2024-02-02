import cv2 as cv
import winsound

def take_camera():
    cap = cv.VideoCapture(0)
    if cap.isOpened():
        
        winsound.PlaySound("C:/Users/tjral/yolov5/voice/count.wav", winsound.SND_ALIAS)  
        ret, img = cap.read()
        winsound.PlaySound("C:/Users/tjral/yolov5/voice/camera.wav", winsound.SND_ALIAS)   
        
        if not ret:
            print("Can't read camera")
            winsound.PlaySound("C:/Users/tjral/yolov5/voice/retry.wav", winsound.SND_ALIAS)
        else:
            print("Read camera")
            img_captured = cv.imwrite('img_captured.png', img)
            
    else:
        print("Camera open failed")
        winsound.PlaySound("C:/Users/tjral/yolov5/voice/nocamera.wav", winsound.SND_ALIAS)
        

    cap.release()
    cv.destroyAllWindows()