"""
Usage: 
python img2video.py path/to/image/dir(/*.jpg, *.jpg ...)
"""
import cv2
import glob, os, sys


def main(file):

    images = glob.glob(os.path.join(file, '*.jpg'))
    images.sort()

    sample = cv2.imread(images[0])
    h, w, _ = sample.shape

    fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
    video = cv2.VideoWriter(os.path.join(file, 'video.mp4'), fourcc, 2.0, (w, h))

    for i in range(len(images)):
        img = cv2.imread(images[i])
        # img = cv2.resize(img, (640,480))
        video.write(img)

    video.release()

if __name__ == '__main__':
    file = sys.argv[1]
    print(file)
    # sys.exit()
    main(file)
