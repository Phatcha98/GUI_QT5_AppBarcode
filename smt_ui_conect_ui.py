import sys
from PyQt5.QtWidgets import QMessageBox, QApplication
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtMultimedia import QCameraInfo
from pylibdmtx.pylibdmtx import decode
from collections import Counter
from PyQt5.QtCore import QCoreApplication, QTimer, QThread, QTimer
import threading
from datetime import datetime
import numpy as np
import pandas as pd
import psycopg2
from ultralytics import YOLO
import cv2
from queue import Queue
from threading import  Event
from PyQt5 import QtWidgets

from zbar_smt.smt_ui_no1 import Ui_MainWindow
app = QtWidgets.QApplication(sys.argv)
MainWindow = QtWidgets.QMainWindow()

#ip 192.168.100.100
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
        # self.pb_run.setEnabled(False)
        # self.pb_reset.setEnabled(False)
        self.pb_stop.setEnabled(False)
        self.st = 0
        self.clicked = True
        self.runclicked = False
        self.init()
    def init(self):
        self.pb_run.clicked.connect(self.run_button_clicked)
        self.pb_connect.clicked.connect(self.connect_button_clicked)
        self.pb_reset.clicked.connect(self.reset_button_clicked)
        self.pb_stop.clicked.connect(self.stop_button_clicked)
        self.brightness_slider.valueChanged.connect(self.update_brightness)
        self.contrast_slider.valueChanged.connect(self.update_contrast)

        self.lineEdit_lot.textChanged.connect(self.on_lineEdit_lot1_textChanged)
        self.lineEdit_quantity_prd.textChanged.connect(self.on_lineEdit_prd_textChanged)
        self.lineEdit_op_id.textChanged.connect(self.on_lineEdit_opid_textChanged)
        self.lineEdit.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_op_id))
        self.lineEdit_lot.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_quantity_prd)) 
        self.lineEdit_quantity_prd.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_partial_no))
        self.lineEdit_partial_no.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_quantity_tray))

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

    def on_lineEdit_opid_textChanged(self):
        try:
            lineEdit_op_id = self.lineEdit_op_id.text()
            op_id_use = set()  # เพื่อให้มีการเก็บค่าที่ไม่ซ้ำกัน

            # ตรวจสอบว่าความยาวของข้อความถูกต้องหรือไม่
            if len(lineEdit_op_id) >= 15:
                # ดึงข้อมูลจากตำแหน่งที่ต้องการ
                op_id_use.add(lineEdit_op_id[:7])
                op_id_use.add(lineEdit_op_id[8:15])

                # ตรวจสอบว่ามีค่าซ้ำกันหรือไม่ ถ้าไม่มีจะสร้างข้อความใหม่
                if len(op_id_use) == 2:
                    op_id_formatted = ','.join(op_id_use)
                    self.lineEdit_op_id.setFont(QFont("Arial", 18))
                    self.lineEdit_op_id.setText(op_id_formatted)
                    self.lineEdit_op_id.returnPressed.connect(lambda: self.focus_next_line_edit(self.lineEdit_lot))
                else:
                    # กรณีมีค่าซ้ำกัน ไม่ต้องทำอะไร
                    pass
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
        result = line_edit1_text[:9]
        self.lineEdit_lot.setFont(QFont("Arial", 18))
        self.lineEdit_lot.setText(result)

    def stop_button_clicked(self):
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
    
############################################################### Connect ##############################################################################
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

                self.th_connect_lan = QThread()
                self.worker_connect_lan = self.click_con()
                self.worker_connect_lan.moveToThread(self.th_connect_lan)
                self.th_connect_lan.started.connect(self.worker_connect_lan.run)
                self.th_connect_lan.start()

            except Exception as ex:
                print(ex)
        else:
            self.CAMERA_INDEX = None
            print("Selected camera not found in the list.")
        
    def click_con(self):
        ret, img = self.cap.read()
        if ret:
            frame = cv2.resize(img.astype(np.uint8), (640, 480))
            QCoreApplication.processEvents()
            self.update_connect(frame) 

    def toggle_video_source(self, source):
        if source == "connect":
            self.clicked = True
            self.runclicked = False
        elif source == "run":
            self.clicked = False
            self.runclicked = True
    def update_connect(self, frame):
        while self.cap.isOpened():  
            ret,frame_cap = self.cap.read()
            if ret:
                color = cv2.cvtColor(frame_cap,cv2.COLOR_BGR2RGB)
                display_img = cv2.resize(color, (640, 480))
                height, width, channel = display_img.shape
                bytesPerLine = channel * width
                qImg = QImage(display_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                self.pixmap = QPixmap.fromImage(qImg)
                self.vdo_frame.setPixmap(self.pixmap)
                cv2.waitKey(1)    
                   
                if self.runclicked:
                    break

    def show_error_message(self, title, message):
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec_()

########################################################################### RUN ########################################################################
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
                    self.model = YOLO(r'dataset/train4/weights/ink-L_v3.pt')
                    # self.model = YOLO(r'D:\yolo8\dataset\train4\weights\best.pt')
                if comboBox_barcode  == 'Barcode ink (Small size)':
                    self.model = YOLO(r'D:\projects\yolo8\dataset\train4\weights\Ink-S_v7.pt')
                    # self.model = YOLO(r'D:\yolo8\dataset\train4\weights\barcode_smallv4.pt')
                if comboBox_barcode  == 'Barcode ink (long size)':             
                    self.model = YOLO(r'D:\projects\yolo8\dataset\train4\weights\Ink-L-S_v2.pt')
                if comboBox_barcode  == 'Barcode paper':
                    self.model = YOLO(r'D:\projects\yolo8\dataset\train4\weights\paper-M_v1.pt')      
                    # self.model = YOLO(r'D:\yolo8\dataset\train4\weights\barcode-normalv2.pt')               
                if comboBox_barcode  == 'Barcode SUS plate':
                    self.model = YOLO(r'D:\projects\yolo8\dataset\train4\weights\sus-S_v4.pt')

                self.cap = cv2.VideoCapture(self.CAMERA_INDEX)
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.ORIGINAL_WIDTH)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.ORIGINAL_HEIGHT) 
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                
                self.thread1 = threading.Thread(target=self.run)
                self.thread1.start()
                # Start thread for reading QR code
                self.thread = threading.Thread(target=self.read_qrcode)
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

    def run(self):
        self.runclicked = True
        all_counts = []
        while self.cap.isOpened():  
            ret,frame_cap = self.cap.read()
            if ret:      
                results = self.model(frame_cap, stream=False, conf=0.4)
                results_frame = results[0].plot()
                color = cv2.cvtColor(results_frame,cv2.COLOR_BGR2RGB)
                display_img = cv2.resize(color, (640, 480))
                height, width, channel = display_img.shape
                bytesPerLine = channel * width
                qImg = QImage(display_img.data, width, height, bytesPerLine, QImage.Format_RGB888)
                self.pixmap = QPixmap.fromImage(qImg)
                self.vdo_frame.setPixmap(self.pixmap)

                self.q = [frame_cap]
                self.y = [results]

                latest_barcode_count = 0  
                QCoreApplication.processEvents()
                for img, results in zip(self.q, self.y):
                    all_results = [item for sublist in results for item in sublist]
                    all_names = [result.names[0] for result in all_results if result.names]
                    result_counts = Counter(all_names)
                    all_counts.append(result_counts)

                    if 'Barcode' in result_counts:
                        latest_barcode_count = result_counts['Barcode']  

                if latest_barcode_count == 0:
                    self.reset_values()
                
                self.quantity_count_detect.setText(str(latest_barcode_count))
                QCoreApplication.processEvents()
                cv2.waitKey(1)
                if self.clicked:
                    break
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

########################################################################## READ ######################################################################
    def enhance_image(self, image):
        # Apply a Gaussian blur to the image to remove noise
        blur = cv2.GaussianBlur(image, (5, 5), 0)
        # Use a kernel to sharpen the blurred image
        sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        sharpened_image = cv2.filter2D(blur, -3, sharpen_kernel)
        return sharpened_image
    
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

############################################################################### SAVE ###################################################################
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
            # with psycopg2.connect(host="10.17.72.65", port=5432, user="postgres", password="postgres",database="iot") as connection:
            with psycopg2.connect(host="localhost", port=5432, user="postgres", password="postgres",database="postgres") as connection:
                for bar in self.barcode:
                    with connection.cursor() as cursor:
                        cursor.execute("""
                            INSERT INTO smt_fin_sn_record
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
