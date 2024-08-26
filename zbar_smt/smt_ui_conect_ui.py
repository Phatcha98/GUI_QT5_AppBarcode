import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QComboBox, QLineEdit, QPushButton, QFrame, QMessageBox, QSlider, QVBoxLayout, QWidget
from PyQt5.QtGui import QColor,QImage, QPixmap, QFont
from PyQt5.QtMultimedia import QCameraInfo, QCamera
from pylibdmtx.pylibdmtx import decode
from collections import Counter
from PyQt5.QtCore import QCoreApplication, QTimer, Qt, QThread, QTimer, pyqtSignal, QObject
import threading
from datetime import datetime
import numpy as np
import pandas as pd
import psycopg2
from ultralytics import YOLO
from PyQt5 import uic, QtGui
import cv2
from queue import Queue
from threading import Thread, Event
from PyQt5 import QtCore, QtGui, QtWidgets
import psutil

from smt_ui_no1 import Ui_MainWindow
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

class Worker(QObject):
    update_signal = pyqtSignal(float)

    def __init__(self):
        super().__init__()

    def update_cpu_usage(self):
        while True:
            cpu_percentage = psutil.cpu_percent(interval=1)
            self.update_signal.emit(cpu_percentage)
class ObjectDetection(Ui_MainWindow):   
    def __init__(self):
        super().setupUi(MainWindow)
        self.comboBox.addItems([c.description() for c in QCameraInfo.availableCameras()])
        self.q = Queue()
        self.y = Queue()
        self.run_button_enabled = True 
        self.connect_button_enabled = True 
        self.pb_run.clicked.connect(self.run_button_clicked)
        self.pb_connect.clicked.connect(self.connect_button_clicked)
        self.cap = None
        self.sql_update_enabled = True
        self.stop_event = Event()
        self.timer = None
        self.thread = None
        self.msg_box = QMessageBox()
        self.lineEdit_lot.textChanged.connect(self.on_lineEdit_lot1_textChanged)
        self.lineEdit_quantity_prd.textChanged.connect(self.on_lineEdit_prd_textChanged)
        self.lineEdit.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_op_id))
        self.lineEdit_op_id.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_lot))
        self.lineEdit_lot.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_quantity_prd))
        self.lineEdit_quantity_prd.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_partial_no))
        self.lineEdit_partial_no.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_quantity_tray))
        self.pb_run.setEnabled(False)
        self.pb_reset.setEnabled(False)
        self.pb_stop.setEnabled(False)
        self.st = 0
        self.init()
    def init(self):
        self.pb_run.clicked.connect(self.run_button_clicked)
        self.pb_connect.clicked.connect(self.connect_button_clicked)
        self.pb_reset.clicked.connect(self.reset_button_clicked)
        self.pb_stop.clicked.connect(self.stop_button_clicked)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        self.contrast_slider.valueChanged.connect(self.update_contrast)

    def update_brightness(self, value):
        if self.cap is not None and self.cap.isOpened():
            # Convert value to the range expected by OpenCV (-100 to 100)
            brightness_value = value
            contrast_value = self.contrast_slider.value()
            self.cap.set(cv2.CAP_PROP_CONTRAST, contrast_value)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
            self.label_Brightness_bar_lv.setText(str(value))

    def update_contrast(self, value):
        if self.cap is not None and self.cap.isOpened():
            # Convert value to the range expected by OpenCV (-100 to 100)
            contrast_value = value
            brightness_value = self.brightness_slider.value()
            self.cap.set(cv2.CAP_PROP_CONTRAST, contrast_value)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, brightness_value)
            self.label_Contrast_bar_lv.setText(str(value))

    def focus_next_line_edit(self, next_line_edit):
        next_line_edit.setFocus()

    def on_lineEdit_prd_textChanged(self):
        line_edit1_text_prd = self.lineEdit_quantity_prd.text()
        start_index = line_edit1_text_prd.find("R")
        end_index = line_edit1_text_prd.rfind("C")
        if start_index != -1 and end_index != -1 and start_index < end_index:
            extracted_text = line_edit1_text_prd[start_index:end_index + 1]
            # Extract alphanumeric characters from the extracted text
            extracted_text = ''.join(c for c in extracted_text if c.isalnum())
            if len(extracted_text) >= 9:
                set1 = extracted_text[:4]
                set2 = extracted_text[4:9]
                set3 = extracted_text[9:]
                prd = set1 + "-" + set2 + "-" + set3
                self.lineEdit_quantity_prd.setText(prd)
                if prd == "":
                    self.lineEdit_quantity_prd.clear()

    def on_lineEdit_lot1_textChanged(self):
        line_edit1_text = str(self.lineEdit_lot.text())
        self.result = line_edit1_text[:9]
        self.lineEdit_lot.setText(self.result)
        self.lineEdit_lot.setFont(QFont("Arial", 18))
        if line_edit1_text== "":
            self.lineEdit_quantity_prd.clear()
                        
    def stop_button_clicked(self):
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        self.timer.stop()
        self.st = 1
        self.stop_event.set()
        self.q = []
        self.y = []
        # self.quantity_count_save.setText("0")
        # self.quantity_result_ok_ng_actual.setText("NG")
        # self.quantity_result_ok_ng_actual.setStyleSheet('color: red')
        self.connect_button_clicked()
        self.reset_values() 
    def reset_button_clicked(self):
        self.st = 0
        self.pb_reset.setStyleSheet("color: #FF8C00; background-color: #FFFFF0;")
        self.pb_run.setStyleSheet("background-color: rgb(189, 189, 189);")
        self.pb_connect.setStyleSheet("background-color: rgb(189, 189, 189);")
        self.pb_stop.setStyleSheet("background-color: rgb(189, 189, 189);")
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        self.lineEdit.clear()
        self.lineEdit_quantity_tray.clear()
        self.lineEdit_lot.clear()
        self.lineEdit_op_id.clear()
        self.lineEdit_partial_no.clear()
        self.lineEdit_quantity_prd.clear()
        self.quantity_count_detect.clear()
        self.quantity_count_save.setText("0")
        self.quantity_result_ok_ng_actual.setText("NG")
        self.quantity_result_ok_ng_actual.setStyleSheet('color: red')
        self.vdo_frame.clear()
        self.stop_event.set()
        self.q = []
        self.y = []
        self.reset_values() 
    
    def connect_button_clicked(self):
        if self.st == 0 :
            self.pb_connect.setEnabled(False)
            self.pb_run.setEnabled(True)
            self.pb_reset.setEnabled(False)
            self.pb_stop.setEnabled(False)
            self.pb_connect.setStyleSheet("color: #0000FF; background-color: #F0FFFF;")
            self.pb_reset.setStyleSheet("background-color: rgb(189, 189, 189);")
            self.pb_run.setStyleSheet("background-color: rgb(189, 189, 189);")
            self.pb_stop.setStyleSheet("background-color: rgb(189, 189, 189);")
        elif self.st == 1:
            self.pb_connect.setEnabled(True)
            self.pb_run.setEnabled(True)
            self.pb_reset.setEnabled(True)
            self.pb_stop.setEnabled(False)
            self.pb_stop.setStyleSheet("color: #FF0000; background-color: #FFF5EE;")
            self.pb_run.setStyleSheet("background-color: rgb(189, 189, 189);")
            self.pb_connect.setStyleSheet("background-color: rgb(189, 189, 189);")
            self.pb_reset.setStyleSheet("background-color: rgb(189, 189, 189);")
        selected_camera_description = self.comboBox.currentText()
        selected_camera = next((c for c in QCameraInfo.availableCameras() if c.description() == selected_camera_description), None)
        if selected_camera is not None:
            try:
                camera_index = QCameraInfo.availableCameras().index(selected_camera)
                self.CAMERA_INDEX = camera_index
                self.ORIGINAL_WIDTH = (100 / 100) * 7700
                self.ORIGINAL_HEIGHT = (100 / 100) * 3600
                self.cap = cv2.VideoCapture(self.CAMERA_INDEX, cv2.CAP_DSHOW)
                print("Camera opened:", self.cap.isOpened())
                
                if not self.cap.isOpened():
                    error_message = "Failed to open the camera. Please make sure it is connected and not in use by another application."
                    self.show_error_message("Error connecting to the camera", error_message)
                    # self.set_connect_button_state(True)
                    return

                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ORIGINAL_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.ORIGINAL_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

                self.timer = QTimer()
                self.timer.timeout.connect(lambda: self.update_connect())
                self.timer.start()

            except Exception as ex:
                error_message = str(ex)
                if "MSMF" in error_message:
                    self.show_error_message("Error connecting to the camera", "MSMF: Can't grab frame.")
                else:
                    # Show a generic error message for other exceptions
                    self.show_error_message("Error connecting to the camera", error_message)
                    print(f"Error connecting to the camera: {error_message}")
        else:
            self.CAMERA_INDEX = None
            print("Selected camera not found in the list.")

    def show_error_message(self, title, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    def run_button_clicked(self):
        self.pb_run.setStyleSheet("color: #00FF7F; background-color: #F0FFF0;")
        self.pb_connect.setStyleSheet("background-color: rgb(189, 189, 189);")
        self.pb_reset.setStyleSheet("background-color: rgb(189, 189, 189);")
        self.pb_stop.setStyleSheet("background-color: rgb(189, 189, 189);")
        self.pb_connect.setEnabled(False)
        self.pb_run.setEnabled(False)
        self.pb_reset.setEnabled(False)
        self.pb_stop.setEnabled(True)        
        self.comboBox_barcode_text = self.comboBox_barcode.currentText()
        selected_camera_description = self.comboBox.currentText()
        selected_camera = next((c for c in QCameraInfo.availableCameras() if c.description() == selected_camera_description), None)
        if selected_camera is not None:
            camera_index = QCameraInfo.availableCameras().index(selected_camera)
            comboBox_barcode  = str(self.comboBox_barcode_text)
            if selected_camera is not None:
                self.CAMERA_INDEX = camera_index
                self.ORIGINAL_WIDTH = (100 / 100) * 7700
                self.ORIGINAL_HEIGHT = (100 / 100) * 3600
                if comboBox_barcode  == 'Barcode ink (Large size)':
                    self.model = YOLO(r'D:\yolo8\dataset\train4\weights\barcode-normalv2.pt')
                    # self.model = YOLO(r'D:\yolo8\dataset\train4\weights\best.pt')
                if comboBox_barcode  == 'Barcode ink (Small size)':
                    self.model = YOLO(r'D:\yolo8\dataset\train4\weights\barcode_smallv6.pt')
                    # self.model = YOLO(r'D:\yolo8\dataset\train4\weights\barcode_smallv4.pt')
                if comboBox_barcode  == 'Barcode ink (long size)':             
                    self.model = YOLO(r'D:\yolo8\dataset\train4\weights\longv1.pt')
                if comboBox_barcode  == 'Barcode paper':
                    self.model = YOLO(r'D:\yolo8\dataset\train4\weights\barcode_black_pagev3.pt')      
                    # self.model = YOLO(r'D:\yolo8\dataset\train4\weights\barcode-normalv2.pt')               
                if comboBox_barcode  == 'Barcode SUS plate':
                    self.model = YOLO(r'D:\yolo8\dataset\train4\weights\susv1.pt')

                self.cap = cv2.VideoCapture(self.CAMERA_INDEX)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ORIGINAL_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.ORIGINAL_HEIGHT)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                self.timer = QTimer()
                self.timer.timeout.connect(lambda: self.run()) 
                self.thread = threading.Thread(target=self.read_qrcode) 
                QCoreApplication.processEvents()
                self.timer.start()
                self.thread.start()
            else:
                self.CAMERA_INDEX = None
                print("Invalid camera index:", self.CAMERA_INDEX)
                return

    def closeEvent(self, event):
        # Release the camera when the application is closed
        if self.cap and self.cap.isOpened():
            self.cap.release()
        event.accept()
            
    def update_connect(self):
        ret, img = self.cap.read()
        if ret:
            img = cv2.convertScaleAbs(img, alpha=1.9, beta=4)
            img_flipped = cv2.flip(img, -1) 
            frame = cv2.resize(img_flipped, (1027, 804))
            self.display_frame(frame)
            
    def run(self): 
        ret, img = self.cap.read()
        all_counts = []
        self.comboBox_barcode_text_1 = self.comboBox_barcode.currentText()
        comboBox_barcode_1  = str(self.comboBox_barcode_text_1)
        if ret:
            if comboBox_barcode_1 == 'Barcode ink (Small size)' or comboBox_barcode_1 == 'Barcode ink (long size)':
                part_width = int(self.ORIGINAL_WIDTH / 2)
                part_height = int(self.ORIGINAL_HEIGHT / 2)

                crop1 = img[0:int(part_height), 0:int(part_width)][::-1, ::-1]
                crop2 = img[0:int(part_height), int(part_width):][::-1, ::-1]
                crop3 = img[int(part_height):, 0:int(part_width)][::-1, ::-1]
                crop4 = img[int(part_height):, int(part_width):][::-1, ::-1]

                # beta1 = 3
                # alpha1 = 2.3
                img1 = cv2.convertScaleAbs(crop1, alpha=1.9, beta=10)
                results1 = self.model(img1, stream=False, conf=0.6)
                results_frame1 = results1[0].plot()

                img2 = cv2.convertScaleAbs(crop2, alpha=1.9, beta=10)
                results2 = self.model(img2, stream=False, conf=0.6)
                results_frame2 = results2[0].plot()

                img3 = cv2.convertScaleAbs(crop3, alpha=1.9, beta=10)
                results3 = self.model(img3, stream=False, conf=0.6)
                results_frame3 = results3[0].plot()

                img4 = cv2.convertScaleAbs(crop4, alpha=1.9, beta=10)
                results4 = self.model(img4, stream=False, conf=0.6)
                results_frame4 = results4[0].plot()

                row1 = np.concatenate((results_frame4, results_frame3), axis=1)
                row2 = np.concatenate((results_frame2, results_frame1), axis=1)
                results_combined = np.concatenate((row1, row2), axis=0)
                print("YOLO results shape results_combined:", results_combined.shape)
                self.q= [img1, img2, img3, img4]
                self.y= [results1, results2, results3, results4]
                for img, results in zip(self.q, self.y):
                    # Assuming self.y is a list of lists
                    all_results = [item for sublist in results for item in sublist]
                    # Extract the "names" attribute from each Results object
                    all_names = [result.names[0] for result in all_results if result.names]
                    result_counts = Counter(all_names)
                    all_counts.append(result_counts)
                                # Combine counts for all sub-images
                total_counts = Counter()
                for counts in all_counts:
                    total_counts += counts
                self.count_for_barcode = total_counts['Barcode']
                self.quantity_count_detect.setText(str(self.count_for_barcode))
                QCoreApplication.processEvents()
                if self.count_for_barcode == "0" or self.count_for_barcode == 0:
                    self.reset_values()
                self.frame_results = cv2.resize(results_combined.astype(np.uint8), (1027, 804))
                QCoreApplication.processEvents()
                self.display_frame1(self.frame_results)

            if comboBox_barcode_1 == 'Barcode ink (Large size)' or comboBox_barcode_1 == 'Barcode paper' or comboBox_barcode_1 == 'Barcode SUS plate':
                # self.gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                part_width = int(self.ORIGINAL_WIDTH / 3)
                part_height = int(self.ORIGINAL_HEIGHT)

                crop1 = img[0:int(part_height), 0:int(part_width)][::-1, ::-1]
                crop2 = img[0:int(part_height), int(part_width):2*int(part_width)][::-1, ::-1]
                crop3 = img[0:int(part_height), 2*int(part_width):][::-1, ::-1]

                # beta1 = 3
                # alpha1 = 2.3
                img1 = cv2.convertScaleAbs(crop1, alpha=1.9, beta=10)
                results1 = self.model(img1, stream=False, conf=0.55)
                results_frame1 = results1[0].plot()

                img2 = cv2.convertScaleAbs(crop2, alpha=1.9, beta=10)
                results2 = self.model(img2, stream=False, conf=0.55)
                results_frame2 = results2[0].plot()

                img3 = cv2.convertScaleAbs(crop3, alpha=1.9, beta=10)
                results3 = self.model(img3, stream=False, conf=0.55)
                results_frame3 = results3[0].plot()

                # Concatenate results for display
                results_combined = np.concatenate((results_frame3, results_frame2, results_frame1), axis=1)

                print("YOLO results shape results_combined:", results_combined.shape)
                self.q = [img1, img2, img3]
                self.y = [results1, results2, results3]

                for img, results in zip(self.q, self.y):
                    # Assuming self.y is a list of lists
                    all_results = [item for sublist in results for item in sublist]
                    # Extract the "names" attribute from each Results object
                    all_names = [result.names[0] for result in all_results if result.names]
                    result_counts = Counter(all_names)
                    all_counts.append(result_counts)

                # Combine counts for all sub-images
                total_counts = Counter()
                for counts in all_counts:
                    total_counts += counts
                self.count_for_barcode = total_counts['Barcode']
                self.quantity_count_detect.setText(str(self.count_for_barcode))
                QCoreApplication.processEvents()
                if self.count_for_barcode == "0" or self.count_for_barcode == 0:
                    self.reset_values()
                self.frame_results = cv2.resize(results_combined.astype(np.uint8), (1027, 804))
                QCoreApplication.processEvents()
                self.display_frame1(self.frame_results)

    def reset_values(self):
        self.barcode.clear()
        self.barcode_set.clear()
        self.barcode_num = 0
        # self.quantity_count_save.setText("0")
        self.quantity_result_ok_ng_actual.setText("NG")
        self.quantity_result_ok_ng_actual.setStyleSheet('color: red')
        self.y = []
        self.q = []
        self.stop_event.clear()

    def display_frame(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_image = q_image.rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.vdo_frame.setPixmap(pixmap)

    def display_frame1(self, frame):
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        q_image = q_image.rgbSwapped()
        pixmap = QPixmap.fromImage(q_image)
        self.vdo_frame.setPixmap(pixmap)

    def enhance_image(self, image):
        # แยกภาพเป็นช่องสี B, G, R
        b, g, r = cv2.split(image)

        # ประมวลผลแต่ละช่องสีแยกกัน
        for channel in [b, g, r]:
            # คำนวณ histogram ของช่องสี
            hist, _ = np.histogram(channel.flatten(), 256, [0, 256])
            # คำนวณ cumulative distribution function (CDF) ของ histogram
            cdf = hist.cumsum()
            cdf_normalized = cdf / cdf.max()
            # กำหนดค่าพิกเซลที่ต่ำที่สุดและสูงที่สุดโดยไม่รวมข้อมูลที่น้อยที่สุด
            min_val = np.argmax(cdf_normalized > 0.1)  # ของพิกเซลที่มืดที่สุด
            max_val = np.argmax(cdf_normalized > 0.90)  # ของพิกเซลที่สว่างที่สุด
            # ปรับ threshold ตามต้องการ (ลดค่า threshold)
            min_val = max(0, min_val - 1)
            max_val = min(255, max_val + 1)
            alpha = 1.3  # ลองปรับค่านี้
            beta = 20 # ปรับค่าตามต้องการ
            # นำ alpha และ beta มาปรับค่าความสว่างและความคมชัดของช่องสี
            channel[:] = cv2.convertScaleAbs(channel, alpha=alpha, beta=beta)
        # รวมช่องสีกลับเข้าด้วยกันเพื่อได้ภาพที่ปรับค่าความสว่างและความคมชัดแล้ว
        enhanced_image = cv2.merge([b, g, r])
        return enhanced_image

    def read_qrcode(self):
        # self.check_data()
        self.barcode = []
        self.barcode_set = set()
        self.barcode_num = 0
        while not self.stop_event.is_set():
            try:
                if self.q and self.y:
                    frames = self.q
                    yolo_results_list = self.y

                    if not frames or not yolo_results_list:
                        print("Frames or YOLO results are empty.")
                        continue

                    for i, (frame, yolo_results) in enumerate(zip(frames, yolo_results_list)):
                        try:
                            for yolo in yolo_results:
                                if yolo.boxes.xyxy is not None and len(yolo.boxes.xyxy) > 0:
                                    for box in yolo.boxes.xyxy:
                                        x_min, y_min, x_max, y_max = box.int().tolist()
                                        comboBox_barcode  = str(self.comboBox_barcode_text)
                                        if comboBox_barcode  == 'Barcode ink (Large size)':
                                            roi = frame[y_min+8:y_max-7, x_min+10:x_max-8] #Barcode Data Matrix Normal
                                            # roi = frame[y_min-8:y_max+8, x_min-8:x_max+8]
                                        if comboBox_barcode  == 'Barcode ink (Small size)':
                                            roi = frame[y_min+4:y_max-4, x_min+8:x_max-11] #Barcode Data Matrix Small
                                            # roi = frame[y_min:y_max, x_min:x_max] 
                                        if comboBox_barcode  == 'Barcode ink (long size)':
                                            # roi = frame[y_min+16:y_max-5, x_min+8:x_max-8] #Barcode Data Matrix Longs
                                            roi = frame[y_min+3:y_max-3, x_min-3:x_max+3] 

                                        if comboBox_barcode  == 'Barcode paper':
                                            roi = frame[y_min+9:y_max-9, x_min+8:x_max-8] #Barcode Data Matrix On Black Page
                                        if comboBox_barcode  == 'Barcode SUS plate':
                                            roi = frame[y_min:y_max, x_min-9:x_max+13] #Barcode Data Matrix On SUS(Steel) Page
                                        # Enhance the ROI (Region of Interest)
                                        enhanced_roi = self.enhance_image(roi)
                                        # Save the enhanced ROI 
                                        cv2.imwrite(f"D:/yolo8/dataset/temp/test_{i}.jpg", enhanced_roi)
                
                                        decoded = decode(enhanced_roi, max_count=1)
                                        if decoded:
                                            decoded_value = decoded[0][0].decode('utf-8')
                                            # Check if the decoded value starts with 'T' or 'J'
                                            if decoded_value.startswith('T') or decoded_value.startswith('J'):
                                            # if decoded_value.startswith('T'):
                                                if decoded_value not in self.barcode_set:
                                                    self.barcode_set.add(decoded_value)
                                                    self.barcode.append(decoded_value)
                                                    self.update_quantity_count()
                                                    QCoreApplication.processEvents()
                                            else:
                                                print(f"Ignoring barcode {decoded_value} as it doesn't start with 'T'")
                                                continue
                                        else:
                                            print(f"No decoding result for the barcode in frame {i}")
                                        
                                    self.save_data()
                                else:
                                    print(f"No boxes detected in frame {i}")

                        except IndexError as ex:
                            print(f"Error in frame {i}: {ex}")

                    # Add the following line to update quantity count after processing each batch
                    self.update_quantity_count()
                QThread.msleep(10)
            except Exception as ex:
                print(f"Error: {ex}")


    def update_quantity_count(self):
        if self.barcode:
            self.barcode_num = len(self.barcode)
            self.quantity_count_save.setText(str(self.barcode_num))
        else:
            self.quantity_count_save.setText("0")

    def save_data(self):
        time1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        mc_value = str(self.lineEdit.text())
        lot_value = str(self.lineEdit_lot.text())
        op_id = str(self.lineEdit_op_id.text())
        partial_no = str(self.lineEdit_partial_no.text())
        quantity_value = str(self.lineEdit_quantity_tray.text()) 
        prd_fi = str(self.lineEdit_quantity_prd.text())
        # print(prd_fi) 
        # quantity_result =  str(self.lineEdit_quantity.text())
        max_id = None
        update_flg = None
        barcode_num = self.barcode_num
        # print(self.barcode)
        if str(barcode_num) == str(quantity_value) and str(lot_value) != " ":
            with psycopg2.connect(host="10.17.72.65", port=5432, user="postgres", password="postgres",database="iot") as connection:
                for bar in self.barcode:
                    with connection.cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO test.smt_fin_sn_record
                            (sn,"time",machine_no,lot_no,op_id,partial_no,total_pcs,max_id,update_flg)
                            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (sn)
                            DO UPDATE SET time = EXCLUDED.time,
                                        machine_no = EXCLUDED.machine_no,
                                        lot_no = EXCLUDED.lot_no,
                                        op_id = EXCLUDED.op_id,
                                        partial_no = EXCLUDED.partial_no,
                                        total_pcs = EXCLUDED.total_pcs,
                                        max_id = EXCLUDED.max_id,
                                        update_flg = EXCLUDED.update_flg
                        """, (bar, time1, mc_value, lot_value, op_id, partial_no, quantity_value, max_id, update_flg))
                        print(f"{bar} to data table sql")

                self.barcode.clear() 
                barcode_num = 0
                self.stop_event.clear()
                self.quantity_result_ok_ng_actual.setText("OK")
                self.quantity_result_ok_ng_actual.setStyleSheet('color: green')
        else:
            self.quantity_result_ok_ng_actual.setText("NG")
            self.quantity_result_ok_ng_actual.setStyleSheet('color: red')
    
if __name__ == "__main__":
    myapp=ObjectDetection()
    MainWindow.show()
    sys.exit(app.exec_())