import cv2
import numpy as np
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip


total_color = 9
line_size = 7
blur_value = 7




def edge_mask(img, line_size, blur_value):
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  gray_blur = cv2.medianBlur(gray, blur_value)
  edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, line_size, blur_value)
  return edges


def color_quantization(img, k):
  data = np.float32(img).reshape((-1, 3))
  criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)

#  K-Means
  ret, label, center = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
  center = np.uint8(center)
  result = center[label.flatten()]
  result = result.reshape(img.shape)
  return result

def image_to_cartoon(image):
  edges = edge_mask(image, line_size, blur_value)
  image = color_quantization(image, total_color)
  blurred = cv2.bilateralFilter(image, d=7, sigmaColor=200,sigmaSpace=200)
  cartoon = cv2.bitwise_and(blurred, blurred, mask=edges)
  return cartoon
def vid_to_cartoon(vid):

  fps = int(vid.get(cv2.CAP_PROP_FPS))
  width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH ))
  height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT ))
  size = (width,height)
  success,image = vid.read()
  count = 275
  out = cv2.VideoWriter('end.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
  while success:
    out.write(image_to_cartoon(image))
    success,image = vid.read()
    count += 1
    print(count)
  out.release()

vid_to_cartoon((cv2.VideoCapture('v.mp4')))

def changefps(vid,fpssend):

  fps = int(vid.get(cv2.CAP_PROP_FPS))
  width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH ))
  height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT ))
  size = (width,height)
  success,image = vid.read()
  count = 275
  out = cv2.VideoWriter('end1.mp4',cv2.VideoWriter_fourcc(*'mp4v'), fpssend, size)
  while success:
    out.write(image)
    success,image = vid.read()
    count += 1
    print(count)
  out.release()
#changefps(cv2.VideoCapture('end.mp4'),25)
