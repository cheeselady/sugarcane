import argparse, sys, time, os, datetime
from pathlib import Path
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box, save_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
from utils.billet_length2 import lengthCalc2
import numpy as np
from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QPushButton, QMessageBox
from PyQt5 import QtCore, QtGui, QtWidgets



# keep these line to remain defalt settings
parser = argparse.ArgumentParser()
# parser.add_argument('--weights', nargs='+', type=str, default='yolov7x.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.6, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='inference', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
opt = parser.parse_args()  

class MainWindow(QMainWindow):
    text_update = pyqtSignal(str)  # use for textbox
    capoff = 1  # this is to control the start button from activating during detection
    
    def __init__(self):
        super().__init__()
        # Initialize the UI
        self.initUI()
        # Set up the camera
        self.cap = None
        self.timer = None        
        self.flag = 0 # this is to stop the detection via camera

    def initUI(self):
        # Set up the window
        self.setGeometry(400, 100, 1050, 900)
        self.setWindowTitle("Automated Sugarcane Billet Detection and Quality Assessment System ")
        
        # Create a button and connect it to the detect function
        self.detect_button = QPushButton("Start detect", self)
        self.detect_button.setGeometry(700, 50, 150, 55)
        font = QtGui.QFont()
        font.setPointSize(10)
        self.detect_button.setFont(font)
        self.detect_button.clicked.connect(self.start_detection)
        
        # create stop button to stop detection from camera
        self.stop_button = QPushButton('Stop detect', self)
        self.stop_button.setGeometry(860, 50, 150, 55)
        self.stop_button.setFont(font)
        self.stop_button.clicked.connect(self.stop_detection)

        # Set up the frame for displaying the camera iamge with inference result
        self.label = QLabel(self)
        self.label.setGeometry(50, 50, 630, 480)
        self.label.setFrameShape(QtWidgets.QFrame.Box)
        self.label.setFrameShadow(QtWidgets.QFrame.Raised)
        self.label.setLineWidth(3)
        
        _translate = QtCore.QCoreApplication.translate
        
        self.frame1 = QtWidgets.QFrame(self)
        self.frame1.setGeometry(QtCore.QRect(700, 145, 310, 215))
        self.frame1.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame1.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame1.setLineWidth(2)
        self.frame1.setObjectName("frame")
        self.label1st = QtWidgets.QLabel(self)
        self.label1st.setGeometry(QtCore.QRect(710, 115, 250, 20))
        self.label1st.setObjectName("label")
        font.setBold(True)
        self.label1st.setText(_translate("MainWindow", "First Stage Detection"))
        self.label1st.setFont(font)
        font.setBold(False)
     
        # drop down list for chosing source
        self.input_label = QtWidgets.QLabel(self)
        self.input_label.setGeometry(QtCore.QRect(720, 210, 120, 40))
        self.input_label.setFont(font)
        self.input_label.setObjectName("input_label")
        self.comboBox = QtWidgets.QComboBox(self)
        self.comboBox.setGeometry(QtCore.QRect(850, 210, 130, 40))
        self.comboBox.setFont(font)
        self.comboBox.setObjectName("comboBox")
        self.comboBox.addItem("")
        self.comboBox.addItem("")
        self.comboBox.setItemText(0, _translate("MainWindow", "camera"))
        self.comboBox.setItemText(1, _translate("MainWindow", "images/video"))
        self.input_label.setText(_translate("MainWindow", "Input Source"))
        self.comboBox.activated.connect(self.source)
        self.comboBox.setCurrentIndex(0)
        self.comboBox.currentIndexChanged.connect(self.updatePathLineEdit)
        
        # drop down list for chosing weights 
        self.weight_label = QtWidgets.QLabel(self)
        self.weight_label.setGeometry(QtCore.QRect(720, 155, 125, 40))
        self.weight_label.setFont(font)
        self.weight_label.setObjectName("input_label")
        self.weight_comboBox = QtWidgets.QComboBox(self)
        self.weight_comboBox.setGeometry(QtCore.QRect(850, 155, 130, 40))
        self.weight_comboBox.setFont(font)
        self.weight_comboBox.setObjectName("comboBox")
        self.weight_comboBox.addItem("")
        self.weight_comboBox.addItem("")
        self.weight_comboBox.addItem("")
        self.weight_comboBox.addItem("")
        _translate = QtCore.QCoreApplication.translate
        self.weight_comboBox.setItemText(0, _translate("MainWindow", "yolov7.pt"))
        self.weight_comboBox.setItemText(1, _translate("MainWindow", "billets.pt"))
        self.weight_comboBox.setItemText(2, _translate("MainWindow", "quality.pt"))
        self.weight_comboBox.setItemText(3, _translate("MainWindow", "eye&node.pt"))
        self.weight_label.setText(_translate("MainWindow", "Weight Selection"))
        self.weight_comboBox.activated.connect(self.weight_comboBox.currentText)   
        
        # Create the label and line edit for the path input
        self.pathLabel = QtWidgets.QLabel(self)
        self.pathLabel.setGeometry(QtCore.QRect(720,260,100, 50))
        self.pathLabel.setFont(font)   
        self.pathLabel.setText(_translate("MainWindow", "Source path:"))
        self.pathLineEdit = QtWidgets.QLineEdit(self)
        self.pathLineEdit.setFont(font)          
        self.pathLineEdit.setGeometry(QtCore.QRect(820, 270, 160, 31))
        self.pathLineEdit.setText("N/A for camera")
        self.pathLineEdit.setReadOnly(True)
      
        self.checkBox = QtWidgets.QCheckBox(self)
        self.checkBox.setGeometry(QtCore.QRect(720, 320, 131, 31))
        self.checkBox.setFont(font)
        self.checkBox.setObjectName("checkBox")     
        self.checkBox.setText(_translate("MainWindow", "Save Labels"))           
        
        self.checkBox2 = QtWidgets.QCheckBox(self)
        self.checkBox2.setGeometry(QtCore.QRect(870, 320, 131, 31))
        self.checkBox2.setFont(font)
        self.checkBox2.setObjectName("checkBox")     
        self.checkBox2.setText(_translate("MainWindow", "Save crops"))          
        
        self.frame = QtWidgets.QFrame(self)
        self.frame.setGeometry(QtCore.QRect(700, 400, 310, 130))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Plain)
        self.frame.setLineWidth(2)
        self.frame.setObjectName("frame")
        self.label2nd = QtWidgets.QLabel(self)
        self.label2nd.setGeometry(QtCore.QRect(710, 370, 250, 20))
        self.label2nd.setObjectName("label")
        font.setBold(True)
        self.label2nd.setText(_translate("MainWindow", "Second Stage Detection"))
        self.label2nd.setFont(font)
        font.setBold(False)
       
        self.checkCropDet = QtWidgets.QCheckBox(self)
        self.checkCropDet.setGeometry(QtCore.QRect(790, 410, 131, 31))
        self.checkCropDet.setFont(font)
        self.checkCropDet.setObjectName("checkBox")     
        self.checkCropDet.setText(_translate("MainWindow", "Crops detect"))  
        
  # ----- 2nd stage save label and save image checkboxes ---- #
      
        self.checkBox2ndlab = QtWidgets.QCheckBox(self)
        self.checkBox2ndlab.setGeometry(QtCore.QRect(720, 490, 131, 31))
        self.checkBox2ndlab.setFont(font)
        self.checkBox2ndlab.setObjectName("checkBox")     
        self.checkBox2ndlab.setText(_translate("MainWindow", "Save Labels"))   
        
        self.checkBox2ndimg = QtWidgets.QCheckBox(self)
        self.checkBox2ndimg.setGeometry(QtCore.QRect(870, 490, 131, 31))
        self.checkBox2ndimg.setFont(font)
        self.checkBox2ndimg.setObjectName("checkBox")     
        self.checkBox2ndimg.setText(_translate("MainWindow", "Save images")) 
        
 # ------ 2nd stage detection selection -----
 
        self.checkBoxQA = QtWidgets.QCheckBox(self)
        self.checkBoxQA.setGeometry(QtCore.QRect(720, 450, 135, 31))
        self.checkBoxQA.setFont(font)
        self.checkBoxQA.setObjectName("checkBox")     
        self.checkBoxQA.setText(_translate("MainWindow", "Quality Assess"))   
        
        self.checkBoxEN = QtWidgets.QCheckBox(self)
        self.checkBoxEN.setGeometry(QtCore.QRect(870, 450, 131, 31))
        self.checkBoxEN.setFont(font)
        self.checkBoxEN.setObjectName("checkBox")     
        self.checkBoxEN.setText(_translate("MainWindow", "Eyes & Nodes"))  
      
    # ----- this section is to add a textBox to print out info from terminal ---------     
    # ----- adapted from computer vision course lab gui script   
        self.textbox = QtWidgets.QTextEdit(self)
        self.textbox.setFont(QtGui.QFont("Arial", 10))
        self.textbox.setGeometry(QtCore.QRect(50, 550, 960, 300))
        self.text_update.connect(self.append_text)
        sys.stdout = self        
        print("Welcome!" )        
    # write and flush is important to handle sys.stdout.write: update text display    
    def write(self, text):
        self.text_update.emit(str(text))
    def flush(self):
        pass    
    def append_text(self, text):  
        cur = self.textbox.textCursor()     # Move cursor to end of text
        cur.movePosition(QtGui.QTextCursor.End) 
        s = str(text)
        while s:
            head,sep,s = s.partition("\n")  # Split line at LF
            cur.insertText(head)            # Insert text at cursor
            if sep:                         # New line if LF
                cur.insertBlock()
        self.textbox.setTextCursor(cur)     # Update visible cursor    
        
     
    def updatePathLineEdit(self, index):
        if index == 0: # Camera
            self.pathLineEdit.setText("N/A for camera")
            self.pathLineEdit.setReadOnly(True)
            # self.button.setEnabled(False)
        else:
            self.pathLineEdit.setText("")
            self.pathLineEdit.setReadOnly(False)
            # self.button.setEnabled(True)           
         
    # setting source            
    def source(self):
        # assign source value
        selected_source = self.comboBox.currentText()
        if selected_source == "camera":
            # Assign a variable for camera source
            source = '0'
            self.cam_source = 1
        elif selected_source == "images/video":
            self.cam_source = 0  # flag used for if print out the info
            # Check if the pathLineEdit is empty
            if self.pathLineEdit.text() == "":
                # Use the default path for images
                source = "inference/images/"
            else:
                # Use the user-entered path
                source = self.pathLineEdit.text()
        else:
            # Handle invalid selection
            print("Invalid source selected.")
        return source 


    def do_detect(self):
        if self.capoff:      # if capturing, disable any function
            with torch.no_grad():
                print("The First Stage Detection starts...\n")
                path = self.detect( self.source(), self.weight_comboBox.currentText() )
                print("The First Stage Detection finished.\n")
                
                if self.checkCropDet.isChecked() and self.checkBoxEN.isChecked():
                    print("The Second Stage Detection for eyes and nodes starts...\n")
                    self.detect(os.path.join(path, "crops"), 'eye&node.pt', img_size=416, no1Det=False)
                    print("The Second Stage Detection for eyes and nodes finished.\n")
                if self.checkCropDet.isChecked() and self.checkBoxQA.isChecked():
                    print("The Second Stage Detection for quality assessment starts...\n")
                    self.detect(os.path.join(path, "crops"), 'quality.pt', img_size=416, no1Det=False)       
                    print("The Second Stage Detection for quality assessment finished.\n")



    def start_detection(self):        
      
        if self.checkCropDet.isChecked() and not self.checkBox2.isChecked():
            msg = QMessageBox()
            msg.setWindowTitle("Warning")
            msg.setText("\"Save crop\" must be checked for second stage detection!")
            msg.setIcon(QMessageBox.Warning)
            x = msg.exec_()        
        else:
            if self.checkCropDet.isChecked() and not (self.checkBoxEN.isChecked() or self.checkBoxQA.isChecked()):
                msg = QMessageBox()
                msg.setWindowTitle("Warning")
                msg.setText("Please choose one or both options for Second Stage Detection!")
                msg.setIcon(QMessageBox.Warning)
                x = msg.exec_()
            else:   
                
                if not self.checkBox.isChecked() and  self.weight_comboBox.currentText()=="billets.pt":
    
                    msg = QMessageBox()
                    msg.setWindowTitle("Warning")
                    msg.setText("Please note that if you do not choose to save labels, the average billet length will not be calculated. Are you sure you want to proceed without this information?")
                    msg.setIcon(QMessageBox.Warning)
                    msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No) 
                    button_clicked = msg.exec_()

                    if button_clicked == QMessageBox.Yes:
                        self.do_detect()
                    elif button_clicked == QMessageBox.No:
                        pass
                else: 
                    self.do_detect()
                     

    def stop_detection(self):        
        self.flag = 0   # this is to stop the detection via camera
        LoadStreams.stop(LoadStreams.__class__)
        self.capoff = 1 # this is to control the start button from activating during detection
    
    def openFolder(self, path):
        # Replace backslashes with forward slashes
        path = str(path)
        path = path.replace("\\", "/")        
        # Check if the path exists
        if not os.path.exists(path):
            self.showError("The path does not exist.")
            return
        # If a folder was selected, open the file explorer at the selected folder
        url = QtCore.QUrl.fromLocalFile(path)
        QtGui.QDesktopServices.openUrl(url)

    def detect(self, input_souce, input_weight, img_size=640, no1Det=True, save_img=False):
        self.flag = 1  # this is to stop the detection via camera
               
        source, weights, view_img, save_txt, imgsz, trace = input_souce, input_weight , opt.view_img, ((no1Det == True and self.checkBox.isChecked()) or (no1Det == False and self.checkBox2ndlab.isChecked() )), img_size, not opt.no_trace
        save_img = not opt.nosave and not source.endswith('.txt') and ( (no1Det == True) or (no1Det == False and self.checkBox2ndimg.isChecked()) ) # save inference images
        webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
            ('rtsp://', 'rtmp://', 'http://', 'https://'))
    
        # Directories
        if no1Det:
            save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        else: 
            save_dir = Path(os.path.join(source, "Crops_detect"+"_"+str(weights)[0:3]))
            (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
        
        # Initialize
        set_logging()
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
    
        # Load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        stride = int(model.stride.max())  # model stride
        imgsz = check_img_size(imgsz, s=stride)  # check img_size
    
        if trace:
            model = TracedModel(model, device, opt.img_size)
    
        if half:
            model.half()  # to FP16
    
        # Second-stage classifier
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n=2)  # initialize
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()
    
        # Set Dataloader
        vid_path, vid_writer = None, None
        if webcam:
            view_img = check_imshow()
            cudnn.benchmark = True  # set True to speed up constant image size inference
            dataset = LoadStreams(source, img_size=imgsz, stride=stride)
            self.capoff = 0  # this is to control the start button from activating during detection
        else:
            dataset = LoadImages(source, img_size=imgsz, stride=stride)
    
        # Get names and colors
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]
    
        # Run inference
        if device.type != 'cpu':
            model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
        old_img_w = old_img_h = imgsz
        old_img_b = 1
    
        t0 = time.time()
        ii = 0 
        
        # Get the current date and time
        now = datetime.datetime.now()        
        # Format the date and time as a string
        current_time = now.strftime("%Y-%m-%d_%H-%M-%S")        
        # Create the file name by appending the current date and time to the file extension
        file_name = os.path.join(save_dir, f"log_{current_time}.txt")
        
        with open( file_name, 'a') as file:
            # create an empty dictionary to store the number of detections per class
            class_counts = {}
            for path, img, im0s, vid_cap in dataset:
                img = torch.from_numpy(img).to(device)
                img = img.half() if half else img.float()  # uint8 to fp16/32
                img /= 255.0  # 0 - 255 to 0.0 - 1.0
                if img.ndimension() == 3:
                    img = img.unsqueeze(0)
        
                # Warmup
                if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
                    old_img_b = img.shape[0]
                    old_img_h = img.shape[2]
                    old_img_w = img.shape[3]
                    for i in range(3):
                        model(img, augment=opt.augment)[0]
        
                # Inference
                t1 = time_synchronized()
                with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
                    pred = model(img, augment=opt.augment)[0]
                t2 = time_synchronized()
        
                # Apply NMS
                pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
                t3 = time_synchronized()
                
                # Apply Classifier
                if classify:
                    pred = apply_classifier(pred, modelc, img, im0s)
        
                # Process detections
                if self.flag:   # use for stop_detection                     
                    for i, det in enumerate(pred):  # detections per image
                        if webcam:  # batch_size >= 1
                            p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
                        else:
                            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
            
                        p = Path(p)  # to Path
                        save_path = str(save_dir / p.name)  # img.jpg
                        txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
                        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
                        imc = im0.copy() if self.checkBox2.isChecked() else im0  # for save_crop
                        if len(det):
                            # Rescale boxes from img_size to im0 size
                            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            
                            # Print results
                            for c in det[:, -1].unique():
                                n = (det[:, -1] == c).sum()  # detections per class
                                s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                                         
                                # add the detection to the dictionary
                                cc = int(c)
                                nn =int(n)
                                if cc not in class_counts:
                                    class_counts[cc] = nn
                                    # print("add new class")
                                    # print(cc)
                                    # print(class_counts)
                                else:
                                    class_counts[cc] += nn
                                    # print("increment class")
                                    # print(class_counts)
                            
                            # Write results
                            for *xyxy, conf, cls in reversed(det):
                                if save_txt:  # Write to file
                                    xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                                    line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                                    with open(txt_path + '.txt', 'a') as f:
                                        f.write(('%g ' * len(line)).rstrip() % line + '\n')
            
                                if save_img or view_img:  # Add bbox to image
                                    label = f'{names[int(cls)]} {conf:.2f}'
                                    plot_one_box(xyxy, im0,  label=label, color=colors[int(cls)], line_thickness=1)
                                
                                # crop all bounding boxes
                                if self.checkBox2.isChecked() and no1Det:
                                    clsname = f'{names[int(cls)]}_{conf:.2f}'#'{conf:.2f}'
                                    filepath = Path(os.path.join(save_dir, 'crops', f'{p.stem}'))
                                    save_one_box(xyxy, imc, clsname, file=filepath, BGR=True) 

                        # Print time (inference + NMS)
                        if self.cam_source and no1Det:
                            print(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS')
                        file.write(f'{s}Done. ({(1E3 * (t2 - t1)):.1f}ms) Inference, ({(1E3 * (t3 - t2)):.1f}ms) NMS'+"\n")
                        
                        # stream result - display in the main window
                        frame = cv2.cvtColor(im0, cv2.COLOR_BGR2RGB)
                        height, width, channel = frame.shape
                        bytesPerLine = 3 * width
                        qImg = QImage(frame.data, width, height, bytesPerLine, QImage.Format_RGB888)
                        # Scale the QImage to fit the label
                        pixmap = QPixmap(qImg)
                        pixmap = pixmap.scaled(self.label.width(), self.label.height(), Qt.KeepAspectRatio)
                        # Display the QImage in the label
                        self.label.setPixmap(pixmap)
                            
                        # Save results (image with detections)
                        if save_img:
                            if dataset.mode == 'image':
                                cv2.imwrite(save_path, im0)
                                # print(f" The image with the result is saved in: {save_path}")
                            else:  # 'video' or 'stream'
                                if vid_path != save_path:  # new video
                                    vid_path = save_path
                                    if isinstance(vid_writer, cv2.VideoWriter):
                                        vid_writer.release()  # release previous video writer
                                    if vid_cap:  # video
                                        fps = vid_cap.get(cv2.CAP_PROP_FPS)
                                        w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                        h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                    else:  # stream
                                        fps, w, h = 30, im0.shape[1], im0.shape[0]
                                        save_path += '.mp4'
                                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                                vid_writer.write(im0)
                else:
                    # Release the vid_writer
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()
                    break
                
                if webcam: 
                    time.sleep(1) # pause loop time to match fps
                    ii += 1    # -ii is the pause loop time
                        
            if save_txt or save_img:
                s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
                # print(f"Results saved to {save_dir}{s}")
            if self.cam_source and no1Det:
                print(f'Done. ({time.time() - t0 - ii:.3f}s)') # -ii is the pause loop time  
            file.write(f'Done. ({time.time() - t0 - ii:.3f}s)'+"\n")
    
            # print the number of detections for each class
            ss = ''
            for cls, count in class_counts.items():
                ss += f"{count} {names[int(cls)]}{'s' * (count > 1)}, "
            print("The total sum of each class:", ss[:-2])  # remove the last comma and space
            file.write(f"The total sum of each class: {ss[:-2]} \n")
            
            # after the label folder is generated, if the weights is for billets detection, cal len
            if weights ==  "billets.pt" and self.checkBox.isChecked():
                paths = str(save_dir)
                paths = paths.replace("\\", "/") 
                billets_avglen = (lengthCalc2(source = paths))
                print(f"The average billets length of the load is: {billets_avglen:.3f} cm \n")
                file.write(f"The average billets length of the load is: {billets_avglen:.3f} cm \n")
            
            # open the folder which saved the inferenced images and labels if label save is clicked
            self.openFolder(save_dir)
            
            return save_dir

if __name__ == '__main__':  
    app = QApplication([])
    window = MainWindow()
    window.show()
    app.exec_()