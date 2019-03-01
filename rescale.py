import cv2
import argparse
from imutils import paths
from os.path import join, basename, splitext
from multiprocessing.pool import ThreadPool
import numpy as np

parser = argparse.ArgumentParser(description="A simple image-resizer, to help cut down on the ram- and disk-murdering effects of "
                                             "unnecessarily high-res images.")

parser.add_argument("-l", "--length", help="Side length of output image", required=True, type=int)
parser.add_argument("-e", "--output_ext", help="The file type to export. (Default .png)", default=".png")
parser.add_argument("-s", "--stretch", help="Stretch images instead of adding solid background when reshaping.", action="store_true")
parser.add_argument("-b", "--background_color", help="Background color to use when not stretching images (Default neutral grey)", type=int, default=127)
parser.add_argument("-i", "--input_dir", help="Input Directory", required=True)
parser.add_argument("-o", "--output_dir", help="Output Directory", required=True)

args = parser.parse_args()

# Make sure extension includes the "."
if args.output_ext[0] != '.':
    args.output_ext = "." + args.output_ext

# Get list of image paths
inputPaths = list(paths.list_images(args.input_dir))

# Create list of output paths
# One-Liner Explanation:
# 1) clips off filename from origonal path
# 2) removes old extension and concatenates on new
# 3) joins with new prefix
outputPaths = [join(args.output_dir, splitext(basename(thisPath))[0] + args.output_ext) for thisPath in inputPaths]
images = []


def resizeAndPad(img, size, padColor=127):
    """ Resizes an image without stretching, by putting black borders on.
        Shamelessly borrowed from https://stackoverflow.com/a/44724368/3544375 """

    h, w = img.shape[:2]
    sh, sw = size

    # interpolation method
    if h > sh or w > sw: # shrinking image
        interp = cv2.INTER_AREA
    else: # stretching image
        interp = cv2.INTER_CUBIC

    # aspect ratio of image
    aspect = w/h  # if on Python 2, you might need to cast as a float: float(w)/h

    # compute scaling and pad sizing
    if aspect > 1: # horizontal image
        new_w = sw
        new_h = np.round(new_w/aspect).astype(int)
        pad_vert = (sh-new_h)/2
        pad_top, pad_bot = np.floor(pad_vert).astype(int), np.ceil(pad_vert).astype(int)
        pad_left, pad_right = 0, 0
    elif aspect < 1: # vertical image
        new_h = sh
        new_w = np.round(new_h*aspect).astype(int)
        pad_horz = (sw-new_w)/2
        pad_left, pad_right = np.floor(pad_horz).astype(int), np.ceil(pad_horz).astype(int)
        pad_top, pad_bot = 0, 0
    else: # square image
        new_h, new_w = sh, sw
        pad_left, pad_right, pad_top, pad_bot = 0, 0, 0, 0

    # set pad color
    if len(img.shape) is 3 and not isinstance(padColor, (list, tuple, np.ndarray)): # color image but only one color provided
        padColor = [padColor]*3

    # scale and pad
    scaled_img = cv2.resize(img, (new_w, new_h), interpolation=interp)
    scaled_img = cv2.copyMakeBorder(scaled_img, pad_top, pad_bot, pad_left, pad_right, borderType=cv2.BORDER_CONSTANT, value=padColor)

    return scaled_img

def processImage(ioPathTuple) :
    """ Processes one image """
    (ip,op) = ioPathTuple
    img = cv2.imread(ip)
    if args.stretch:
        img = cv2.resize(img,(args.length,args.length),cv2.INTER_AREA)
    else:
        img = resizeAndPad(img,(args.length,args.length),args.background_color)
    cv2.imwrite(op,img)

# Multithreading, because I won't wait a second longer than I have to
threadPool = ThreadPool(10)
threadPool.map(processImage,zip(inputPaths,outputPaths))
print("Done!")