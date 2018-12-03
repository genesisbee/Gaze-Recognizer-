import cv2

def rescale_frame(frame,w_percent, h_percent):
    row = frame.shape[0]
    col = frame.shape[1]
    print(row,col)
    height= int(frame.shape[1] * (h_percent)/ 100)
    width = int(frame.shape[0] * (w_percent)/ 100)
    dim = (width,height)
    print(dim)
    return cv2.resize(frame,dim,interpolation=cv2.INTER_AREA)

cap = cv2.VideoCapture("Experiment.mp4")

#Use while loop for testing frame size
# while(cap.isOpened()):
#   ret, frame = cap.read()
#   #width ,height
#   frame = rescale_frame(frame,50,20)
#   cv2.imshow('frame',frame)
#   if cv2.waitKey(20)& 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()
