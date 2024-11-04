import cv2

path = 'face.png'
img = cv2.imread(path, 1)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

while True:
    count = 1
    faces = face_detector.detectMultiScale(img_gray, 1.3, 5)
    for (x, y, w, h) in faces:
        img_face = cv2.resize(img[y + 3 : y + h -3, x + 3: x + w - 3],(70,70))
        cv2.imwrite('imgs/img_face{}.jpg'.format(count),img_face )
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        count += 1
    cv2.imshow("anh", img)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break
cv2.destroyAllWindows()