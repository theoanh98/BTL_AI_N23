from mylib.centroidtracker import CentroidTracker
from mylib.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
from mylib import config, thread
import time, schedule, csv
import numpy as np
import argparse, imutils

import time, dlib, cv2, datetime
from itertools import zip_longest
t0 = time.time()
def run():
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--prototxt", required=False,
		help="path to Caffe 'deploy' prototxt file")
	ap.add_argument("-m", "--model", required=True,
		help="path to Caffe pre-trained model")
	ap.add_argument("-i", "--input", type=str,
		help="path to optional input video file")
	ap.add_argument("-o", "--output", type=str,
		help="path to optional output video file")
	# confidence default 0.4
	ap.add_argument("-c", "--confidence", type=float, default=0.4,
		help="minimum probability to filter weak detections")
	ap.add_argument("-s", "--skip-frames", type=int, default=30,
		help="# of skip frames between detections")
	args = vars(ap.parse_args())
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	#load model
	net = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])
	# Mở video
	if not args.get("input", False):
		print("[INFO] Starting the live stream..")
		vs = VideoStream(config.url).start()
		time.sleep(2.0)
	else:
		print("[INFO] Starting the video..")
		vs = cv2.VideoCapture(args["input"])
	writer = None
	W = None
	H = None
	ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
	trackers = []
	trackableObjects = {}
	totalFrames = 0
	totalDown = 0
	totalUp = 0
	x = []
	empty=[]
	empty1=[]
	fps = FPS().start()
	if config.Thread:
		vs = thread.ThreadingClass(config.url)

	# đọc các frame trong video
	while True:
		frame = vs.read()
		frame = frame[1] if args.get("input", False) else frame
		if args["input"] is not None and frame is None:
			break
		frame = imutils.resize(frame, width = 500)
		rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

		if W is None or H is None:
			(H, W) = frame.shape[:2]

		if args["output"] is not None and writer is None:
			fourcc = cv2.VideoWriter_fourcc(*"MJPG")
			writer = cv2.VideoWriter(args["output"], fourcc, 30,
				(W, H), True)

		status = "Waiting"
		rects = []

		if totalFrames % args["skip_frames"] == 0:    #bỏ qua 30 frame cách mỗi lần detecting
			status = "Detecting"
			trackers = []

			# lấy đốm ảnh từ frame rồi đưa qua mô hình để nhận dạng
			blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
			net.setInput(blob)
			detections = net.forward()

			# trích xuất ngưỡng tin để xem đối tượng có phải con người không
			for i in np.arange(0, detections.shape[2]):
				confidence = detections[0, 0, i, 2]
				if confidence > args["confidence"]:
					idx = int(detections[0, 0, i, 1])
					if CLASSES[idx] != "person":
						continue

					# Tính giới hạn bounding box và chuyển tạo độ vào hình chữ nhật
					box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
					(startX, startY, endX, endY) = box.astype("int")
					tracker = dlib.correlation_tracker()
					rect = dlib.rectangle(startX, startY, endX, endY)
					tracker.start_track(rgb, rect)
					trackers.append(tracker)
		else:
			# lặp qua trackers
			for tracker in trackers:
				# cập nhật trạng thái
				status = "Tracking"
				# lấy vị trí đối tượng
				tracker.update(rgb)
				pos = tracker.get_position()
				# trích xuất tọa độ và thêm vào rects
				startX = int(pos.left())
				startY = int(pos.top())
				endX = int(pos.right())
				endY = int(pos.bottom())
				rects.append((startX, startY, endX, endY))

		# vẽ đường kẻ ngang để đếm
		cv2.line(frame, (0, H // 2), (W, H // 2), (0, 0, 255), 3)
		# thêm rects vào objects chứa các đối tượng đang theo dõi
		objects = ct.update(rects)

		# lặp qua các đối tượng đang theo dõi
		for (objectID, centroid) in objects.items():
			to = trackableObjects.get(objectID, None)
			# Nếu không tồn tại đối tượng đang theo dõi thì thêm vào
			if to is None:
				to = TrackableObject(objectID, centroid)

			# Xác định hướng di chuyển
			else:
				y = [c[1] for c in to.centroids]
				direction = centroid[1] - np.mean(y)
				to.centroids.append(centroid)
				# Đếm
				if not to.counted:
					if direction < 0 and centroid[1] < H // 2:
						totalUp += 1
						empty.append(totalUp)
						to.counted = True
					elif direction > 0 and centroid[1] > H // 2:
						totalDown += 1
						empty1.append(totalDown)

						x = []
						x.append(len(empty1)-len(empty))
						to.counted = True
			trackableObjects[objectID] = to

			# hiển thị ID và centroid
			text = "ID {}".format(objectID)
			cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
				cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

		info = [
		("Exit", totalUp),
		("Enter", totalDown),
		("Status", status),
		]

		info2 = [
		("Total people inside", x),
		]

                # Hiển thị kết quả
		for (i, (k, v)) in enumerate(info):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
		for (i, (k, v)) in enumerate(info2):
			text = "{}: {}".format(k, v)
			cv2.putText(frame, text, (265, H - ((i * 20) + 60)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

		if writer is not None:
		   writer.write(frame)

		# ghi lại quá trình đếm
		if config.Log:
			datetimee = [datetime.datetime.now()]
			d = [datetimee, empty1, empty, x]
			export_data = zip_longest(*d, fillvalue = '')

			with open('Log.csv', 'w', newline='') as myfile:
				wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
				wr.writerow(("End Time", "In", "Out", "Total Inside"))
				wr.writerows(export_data)

		cv2.imshow("Video", frame)
		key = cv2.waitKey(1) & 0xFF
		if key == ord("q"):
			break
		totalFrames += 1
		fps.update()
		if config.Timer:
			# Tự động dừng đếm khi vượt quá 8 giờ
			t1 = time.time()
			num_seconds=(t1-t0)
			if num_seconds > 28800:
				break
	fps.stop()
	print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
	print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
	if writer is not None:
		writer.release()
	if not args.get("input", False):
		vs.stop()
	else:
		vs.release()
	cv2.destroyAllWindows()
# tạo lịch đếm
if config.Scheduler:
	schedule.every().day.at("9:00").do(run)
	while 1:
		schedule.run_pending()
else:
	run()


