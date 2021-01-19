import time
from threading import Thread
from typing import Tuple
import cv2 as cv
import imutils
from imutils.video import FileVideoStream, VideoStream

class BallFinder:
  def __init__(self, video_stream: VideoStream, frame_size: Tuple[int, int] = (320, 240)) -> None:
    super().__init__()
    self.stopped = False
    self.vs = video_stream
    self.frame_size = frame_size

    self.ball_found = False
    self.ball_center = (0, 0)
    self.frame = None
    self.ball_greyscale = None
    self.hsv_mask = None
    self.ball_mask = None
    self.mirror_mask = None

  def process(self):
    while not self.stopped:
      frame = imutils.resize(self.vs.read(), width=self.frame_size[0])
      hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

      b, g, r = cv.split(frame)
      h, s, v = cv.split(hsv)

      ball_greyscale = cv.subtract(r, b)
      hsv_mask = cv.bitwise_and(
        cv.inRange(s, 100, 255),
        cv.bitwise_or(
          cv.inRange(h, 0, 4),
          cv.inRange(h, 175, 180),
        ),
      )

      max_val = cv.minMaxLoc(ball_greyscale, hsv_mask)[1]
      ball_mask = cv.erode(
        cv.bitwise_and(
          cv.inRange(
            ball_greyscale,
            (max_val - 45),
            255,
          ),
          hsv_mask,
        ),
        cv.getStructuringElement(
          cv.MORPH_RECT,
          (5, 5),
        )
      )

      ball_found = False
      ball_center = (0, 0)
      if max_val >= 70:
        moments = cv.moments(ball_mask)

        if moments["m00"] <= 0:
          ball_center = int(moments["m10"]), int(moments["m01"])
        else:
          ball_center = int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])

        if ball_center != (0, 0):
          ball_found = True
      
      self.ball_found = ball_found
      self.ball_center = ball_center if ball_found else (0, 0)

      # Find the mirror center
      mirror_mask = cv.inRange(s, 0, 32)
      self.mirror_mask = mirror_mask

      self.frame = frame
      self.ball_greyscale = ball_greyscale
      self.hsv_mask = hsv_mask
      self.ball_mask = ball_mask

      # time.sleep(0.1)

  def get_frame(self):
    return self.ball_found, self.ball_center, self.frame, self.ball_greyscale, self.hsv_mask, self.ball_mask

  def start(self):
    thread = Thread(target=self.process, args=(), name='BallFinderProcess', daemon=True)
    thread.start()

  def stop(self):
    self.stopped = True

if __name__ == "__main__":
  video_file = "data/VID_20201224_134735.mp4"
  video = FileVideoStream(video_file)
  video.start()
  output_video = cv.VideoWriter('output.avi', cv.VideoWriter_fourcc(*"MJPG"), 20.0, (320, 568))

  finder = BallFinder(video)
  try:
    finder.start()

    while True:
      ball_found, ball_center, frame, ball_greyscale, hsv_mask, ball_mask = finder.get_frame()
      if frame is not None:
        if ball_found:
          print('Ball found at ' + str(ball_center))
          cv.circle(frame, ball_center, 4, (255,0,0), -1)	
        else:
          print('Ball not found')
        
        cv.imshow('Frame', frame)
        cv.imshow('Ball Greyscale', ball_greyscale)
        cv.imshow('HSV Mask', hsv_mask)
        cv.imshow('Ball Mask', ball_mask)
        cv.imshow('Mirror Mask', finder.mirror_mask)

        output_video.write(frame)

      if cv.waitKey(1) & 0xFF == ord('q'):
        break
  except KeyboardInterrupt:
    pass
  finder.stop()
  video.stop()
  output_video.release()
