
import cv2

# current camera
cam_path = "rtsp://admin:Aa83446416!@192.168.50.3:554/rtspstream/channel=0/stream=1"
cap = cv2.VideoCapture(cam_path)

# 獲取攝影機的fps
fps = cap.get(cv2.CAP_PROP_FPS)
print("FPS:", fps)

# 獲取攝影機的尺寸
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print("Size:", width, "x", height)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 30, (int(width), int(height)))
while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        out.write(frame)
        cv2.rectangle(frame, (640, 100), (1111, 900), (214, 217, 8), 1, cv2.LINE_AA)
        cv2.imshow('frame', frame)

        key = cv2.waitKey(1)
        # ESC
        if key == 27:
            break

    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
