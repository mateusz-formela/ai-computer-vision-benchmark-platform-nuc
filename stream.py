import cv2

def get_stream(url):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        raise Exception("Nie można połączyć z kamerą")
    return cap
