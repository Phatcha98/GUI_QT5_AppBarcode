import sys
from PyQt5.QtWidgets import  QMessageBox, QApplication
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtMultimedia import QCameraInfo
from pylibdmtx.pylibdmtx import decode
from collections import Counter
from PyQt5.QtCore import QCoreApplication, QTimer, QThread, QTimer
import threading
from datetime import datetime
import numpy as np
import psycopg2
from ultralytics import YOLO
import cv2
from queue import Queue
from threading import Event
from PyQt5 import QtWidgets
from pypylon import pylon
from collections import Counter

from zbar_smt.smt_ui_no1 import Ui_MainWindow
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

class ObjectDetection(Ui_MainWindow):   
    def __init__(self):
        super().setupUi(MainWindow)
        self.comboBox.addItems([c.description() for c in QCameraInfo.availableCameras()])
        self.q = Queue()
        self.y = Queue()
        self.run_button_enabled = True 
        self.connect_button_enabled = True 
        self.cap = None
        self.sql_update_enabled = True
        self.stop_event = Event()
        self.timer = None
        self.thread = None
        self.msg_box = QMessageBox()
        self.lineEdit_lot.textChanged.connect(self.on_lineEdit_lot1_textChanged)
        self.lineEdit_quantity_prd.textChanged.connect(self.on_lineEdit_prd_textChanged)
        self.lineEdit_op_id.textChanged.connect(self.on_lineEdit_opid_textChanged)
        self.lineEdit.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_op_id))
        self.lineEdit_lot.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_quantity_prd))
        self.lineEdit_quantity_prd.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_partial_no))
        self.lineEdit_partial_no.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_quantity_tray))
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
        try:
            if self.camera.IsOpen():
                self.camera.Brightness.SetValue(value)
                self.label_Brightness_bar_lv.setText(str(value))
        except Exception as ex:
            error_message = str(ex)
            print("Error:", error_message)

    def update_contrast(self, value):
        try:
            if self.camera.IsOpen():
                self.camera.Contrast.SetValue(value)
                self.label_Contrast_bar_lv.setText(str(value))
        except Exception as ex:
            error_message = str(ex)
            print("Error:", error_message)

    def focus_next_line_edit(self, next_line_edit):
        next_line_edit.setFocus()

    def on_lineEdit_opid_textChanged(self):
        try:
            lineEdit_op_id = self.lineEdit_op_id.text()
            if len(lineEdit_op_id)== " ":
                self.lineEdit_op_id.create()
            if len(lineEdit_op_id) >= 7:
                op_id = lineEdit_op_id[:7] + ","
                if len(lineEdit_op_id) >7:
                    op_id += lineEdit_op_id[8:15]
                self.lineEdit_op_id.setFont(QFont("Arial", 18))
                self.lineEdit_op_id.setText(op_id)
                if len(lineEdit_op_id) >12:
                    self.lineEdit_op_id.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_lot))
        except Exception as ex:
            print(ex)

    def on_lineEdit_prd_textChanged(self):
        try:
            line_edit1_text_prd = self.lineEdit_quantity_prd.text()
            parts = line_edit1_text_prd.split(";")[2]
            set1 = parts[2:6]
            set2 = parts[6:11]
            set3 = parts[11]
            prd = f"{set1}-{set2}-{set3}"
            prd = set1 + "-" + set2 + "-" + set3
            self.lineEdit_quantity_prd.setText(prd)
            if prd == " ":
                self.lineEdit_quantity_prd.clear()
        except Exception as ex:
            print(f"Error: {ex}")

    def on_lineEdit_lot1_textChanged(self):
        line_edit1_text = str(self.lineEdit_lot.text())
        self.result = line_edit1_text[:9]
        self.lineEdit_lot.setText(self.result)
        self.lineEdit_lot.setFont(QFont("Arial", 18))
        if line_edit1_text== "":
            self.lineEdit_quantity_prd.clear()
                            
    def stop_button_clicked(self):
        # self.stopdetect = True 
        # self.stopdetect1 = False 
        self.st = 1
        self.stop_event.set()
        self.q = []
        self.y = []
        self.connect_button_clicked()
        self.reset_values()

    def reset_button_clicked(self):
        self.st = 0
        self.pb_reset.setStyleSheet("color: #FF8C00; background-color: #FFFFF0;")
        self.pb_run.setStyleSheet("background-color: rgb(189, 189, 189);")
        self.pb_connect.setStyleSheet("background-color: rgb(189, 189, 189);")
        self.pb_stop.setStyleSheet("background-color: rgb(189, 189, 189);")
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
        self.toggle_video_source("connect")
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

        available_cameras = pylon.TlFactory.GetInstance().EnumerateDevices()
        if len(available_cameras) == 0:
            raise Exception("No Basler camera found.")

        # Open the first available camera
        selected_camera = available_cameras[0]
        self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateDevice(selected_camera))
        # self.camera.Open()
        # self.camera.PixelFormat = 'RGB8'
        # Start grabbing images
        self.camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        self.th_connect_lan = QThread()
        self.worker_connect_lan = self.update_connect1()
        self.worker_connect_lan.moveToThread(self.th_connect_lan)
        self.th_connect_lan.started.connect(self.worker_connect_lan.run)
        self.th_connect_lan.start()

        # self.thread2 = threading.Thread(target=self.update_connect1)
        # QCoreApplication.processEvents()
        # self.thread2.start()

    def update_connect1(self):
        try:
            # conecting to the first available camera
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            # Grabing Continusely (video) with minimal delay
            self.camera.StartGrabbing(pylon.GrabLoop_ProvidedByInstantCamera)
            converter = pylon.ImageFormatConverter()
            # converting to opencv bgr format
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            while self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                if grabResult.GrabSucceeded():
                    # Access the image data
                    image = converter.Convert(grabResult)
                    img = image.GetArray()
                    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    # img_flip = cv2.convertScaleAbs(img, alpha=1.9, beta=10)
                    # img = cv2.flip(img_flip, -1) 
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    display_img = cv2.resize(img, (640, 480))
                    height, width, channel = display_img.shape
                    bytesPerLine = channel * width
                    qImg = QImage(display_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                    self.pixmap = QPixmap.fromImage(qImg)
                    self.vdo_frame.setPixmap(self.pixmap)
                    cv2.waitKey(1)
                    if self.runclicked:
                        break
                    # if self.stopDetect:
                    #     break  
        except Exception as ex:
            error_message = str(ex)
            print("Error:", error_message)

    def toggle_video_source(self, source):
        if source == "connect":
            self.clicked = True
            self.runclicked = False
        elif source == "run":
            self.clicked = False
            self.runclicked = True

    def show_error_message(self, title, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

    def run_button_clicked(self):
        self.toggle_video_source("run")
        self.pb_run.setStyleSheet("color: #00FF7F; background-color: #F0FFF0;")
        self.pb_connect.setStyleSheet("background-color: rgb(189, 189, 189);")
        self.pb_reset.setStyleSheet("background-color: rgb(189, 189, 189);")
        self.pb_stop.setStyleSheet("background-color: rgb(189, 189, 189);")
        self.pb_connect.setEnabled(False)
        self.pb_run.setEnabled(False)
        self.pb_reset.setEnabled(False)
        self.pb_stop.setEnabled(True) 
        self.comboBox_barcode_text = self.comboBox_barcode.currentText()
        comboBox_barcode  = str(self.comboBox_barcode_text)
        self.ORIGINAL_WIDTH = (100 / 100) * 4600
        self.ORIGINAL_HEIGHT = (100 / 100) * 2500    
        if comboBox_barcode  == 'Barcode ink (Large size)':
            self.model = YOLO(r'D:\projects\yolo8\dataset\train4\weights\barcode-normalv3.pt')
            # self.model = YOLO(r'D:\yolo8\dataset\train4\weights\best.pt')
        if comboBox_barcode  == 'Barcode ink (Small size)':
            self.model = YOLO(r'D:\projects\yolo8\dataset\train4\weights\barcode_smallv7.pt')
            # self.model = YOLO(r'D:\yolo8\dataset\train4\weights\barcode_smallv4.pt')
        if comboBox_barcode  == 'Barcode ink (long size)':             
            self.model = YOLO(r'D:\projects\yolo8\dataset\train4\weights\susv2.pt')
        if comboBox_barcode  == 'Barcode paper':
            self.model = YOLO(r'D:\projects\yolo8\dataset\train4\weights\barcode_black_pagev1.pt')      
            # self.model = YOLO(r'D:\yolo8\dataset\train4\weights\barcode-normalv2.pt')               
        if comboBox_barcode  == 'Barcode SUS plate':
            self.model = YOLO(r'D:\projects\yolo8\dataset\train4\weights\susv4.pt')
        # Set up timer to trigger the run function periodically
        self.thread1 = threading.Thread(target=self.run)
        # QCoreApplication.processEvents()
        self.thread1.start()
        # self.timer.timeout.connect(lambda: self.set_connect_button_state(True))
        # Start thread for reading QR code
        self.thread2 = threading.Thread(target=self.read_qrcode)
        # QCoreApplication.processEvents()
        self.thread2.start()
        # QCoreApplication.processEvents()

    def run(self): 
        all_counts = []
        # self.stopdetect1 = True
        # self.stopdetect = False
        try:
            self.camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
            # Grabing Continusely (video) with minimal delay
            self.camera.StartGrabbing(pylon.GrabLoop_ProvidedByInstantCamera)
            converter = pylon.ImageFormatConverter()
            # converting to opencv bgr format
            converter.OutputPixelFormat = pylon.PixelType_BGR8packed
            converter.OutputBitAlignment = pylon.OutputBitAlignment_MsbAligned
            while self.camera.IsGrabbing():
                grabResult = self.camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                # QCoreApplication.processEvents()
                if grabResult.GrabSucceeded():
                    # Access the image data
                    image = converter.Convert(grabResult)
                    img = image.GetArray()
                    
                    # img_flip = cv2.convertScaleAbs(img, alpha=1.9, beta=10)
                    # img = cv2.flip(img_flip, -1)
                    results = self.model(img)
                    results_frame = results[0].plot()
                    img1 = cv2.cvtColor(results_frame, cv2.COLOR_BGR2RGB)
                    self.frame_results = cv2.resize(img1, (640, 480))
                    # QCoreApplication.processEvents()
                    height, width, channel = self.frame_results.shape
                    bytesPerLine = channel * width
                    qImg = QImage(self.frame_results.data, width, height, bytesPerLine, QImage.Format_RGB888)
                    pixmap = QPixmap.fromImage(qImg)
                    self.vdo_frame.setPixmap(pixmap)

                    self.q = [img]
                    self.y = [results]

                    latest_barcode_count = 0  # เริ่มต้นให้ค่าเป็น 0 ก่อนทุกครั้งที่สแกนภาพใหม่
                    QCoreApplication.processEvents()
                    for img, results in zip(self.q, self.y):
                        all_results = [item for sublist in results for item in sublist]
                        all_names = [result.names[0] for result in all_results if result.names]
                        result_counts = Counter(all_names)
                        all_counts.append(result_counts)

                        # เช็คว่ามีการตรวจจับบาร์โค้ดหรือไม่
                        if 'Barcode' in result_counts:
                            latest_barcode_count = result_counts['Barcode']  # ใช้ค่าใหม่เมื่อพบบาร์โค้ดในภาพปัจจุบัน

                    if latest_barcode_count == 0:
                        self.reset_values()
                    
                    self.quantity_count_detect.setText(str(latest_barcode_count))
                    QCoreApplication.processEvents()

                    cv2.waitKey(1)
                    if self.clicked:
                        break

        except Exception as e:
            print("An error occurred:", e)

    def reset_values(self):
        self.barcode.clear()
        self.barcode_set.clear()
        self.barcode_num = 0
        self.quantity_count_save.setText("0")
        self.quantity_result_ok_ng_actual.setText("NG")
        self.quantity_result_ok_ng_actual.setStyleSheet('color: red')
        self.y = []
        self.q = []
        self.stop_event.clear()
    def enhance_image(self, image):
        blur = cv2.GaussianBlur(image, (3, 3), 0)
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(blur, -3, sharpen_kernel)
        grayscale_image = cv2.cvtColor(sharpened_image, cv2.COLOR_BGR2GRAY)
        return grayscale_image
    
    def read_qrcode(self):
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
                                            roi = frame[y_min+8:y_max+3, x_min:x_max+3] #Barcode Data Matrix Normal
                                            # roi = frame[y_min-8:y_max+8, x_min-8:x_max+8]
                                        if comboBox_barcode  == 'Barcode ink (Small size)':
                                            roi = frame[y_min+2:y_max-5, x_min+7:x_max-5] #Barcode Data Matrix Small
                                            # roi = frame[y_min:y_max, x_min:x_max] 
                                        if comboBox_barcode  == 'Barcode ink (long size)':
                                            # roi = frame[y_min+16:y_max-5, x_min+8:x_max-8] #Barcode Data Matrix Longs
                                            roi = frame[y_min+3:y_max-3, x_min-3:x_max+3] 
                                        if comboBox_barcode  == 'Barcode paper':
                                            roi = frame[y_min-3:y_max+9, x_min-9:x_max+9] #Barcode Data Matrix On Black Page
                                        if comboBox_barcode  == 'Barcode SUS plate':
                                            roi = frame[y_min:y_max, x_min-9:x_max+13] #Barcode Data Matrix On SUS(Steel) Page
                                        # Enhance the ROI (Region of Interest)
                                        enhanced_roi = self.enhance_image(roi)
                                        # Save the enhanced ROI 
                                        cv2.imwrite(f"D:/yolo8/dataset/temp/test_{i}.jpg", enhanced_roi)

                                        decoded = decode(enhanced_roi, max_count=1)
                                        # decoded = zxingcpp.read_barcodes(enhanced_roi, max_count=1)
                                        if decoded:
                                            decoded_value = decoded[0][0].decode('utf-8')
                                            # Check if the decoded value starts with 'T' or 'J'
                                            if decoded_value.startswith('T') or decoded_value.startswith('J'):
                                                # Check if the decoded value is not already in the set
                                                if decoded_value not in self.barcode_set:
                                                    self.barcode_set.add(decoded_value)
                                                    self.barcode.append(decoded_value)
                                                    self.update_quantity_count()
                                                    QCoreApplication.processEvents()
                                                    # Process events to keep the GUI responsive
                                            else:
                                                print(f"Ignoring barcode {decoded_value} as it doesn't start with '0'")
                                                continue
                                        else:
                                            # print(f"No decoding result for the barcode in frame {i}")
                                            continue
                                        
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
            self.quantity_count_save.setText(str("0"))

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
        if str(barcode_num-1) == str(quantity_value) and str(lot_value) != " ":
            with psycopg2.connect(host="10.17.72.65", port=5432, user="postgres", password="postgres",database="iot") as connection:
            # with psycopg2.connect(host="localhost", port=5432, user="postgres", password="postgres",database="postgres") as connection:
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
