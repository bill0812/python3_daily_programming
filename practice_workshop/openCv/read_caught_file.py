import cv2
from time import time,sleep
cap = cv2.VideoCapture("2-imshow.mp4")

# # 設定擷取影像的尺寸大小
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
print(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# # 使用 mp4v or h264 or xvid的格式將每個 frame 壓縮編碼
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')

# 建立 VideoWriter 物件，輸出影片至 avi 格式的 container
# 正規 NTSC 規定 29.97 較為順暢
# out = cv2.VideoWriter('1-1-with-imshow-timeconsume.mp4', fourcc, 29.97, (1280, 720))
timeconsume = int(1000/30)
# # start = time()
while(True):
  start = time()
  ret, frame = cap.read()
  if ret == True:
    
    # 寫入影格

    # out.write(frame)

    # 顯示當前捕捉到的視窗
    # 但是若同時要寫入影片檔案的話，這行 imshow 會卡住 frame，但是時間依舊在跑
    # 所以簡單來說，多家這行可能讓錄製影片比正常時間要短，而且更快
    # 實際上跟程式運作起來有關係
    # start = time()
    cv2.imshow('frame',frame)
    
    end = time()
    print("true: " + str(end-start))
    # end = time()
    # print(end-start)
    # sleep(timeconsume)
    if cv2.waitKey(timeconsume) & 0xFF == ord('q'):
      break
  else:
    end = time()
    print("false: " + str(end-start))
    break

# 釋放攝影機
cap.release()

out.release()

# 關閉所有 OpenCV 視窗
cv2.destroyAllWindows()