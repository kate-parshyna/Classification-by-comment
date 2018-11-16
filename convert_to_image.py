import cv2

def get_image(folder, path):
    vidcap = cv2.VideoCapture(folder+path)
    success,image = vidcap.read()
    count = 0
    path_im = folder+"frame%d.jpg" % count
    cv2.imwrite(path_im, image)     # save frame as JPEG file
    success,image = vidcap.read()
    return path_im
