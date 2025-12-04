import sys
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QLabel, QTabWidget, QSlider, QFrame, QSizePolicy, QFileDialog, QMessageBox
)
from PyQt5.QtGui import QFont, QColor, QPalette, QImage, QPixmap
from PyQt5.QtCore import Qt, pyqtSignal

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()
        plt.tight_layout()
        
        self.axes.set_xticks([])
        self.axes.set_yticks([])
        self.axes.set_frame_on(False)
        self.draw()
        
    def clear(self):
        self.axes.clear()
        self.draw()
        
    def show_image(self, img):
        self.axes.clear()
        if len(img.shape) == 2:  
            self.axes.imshow(img, cmap='gray')
        else:  
            self.axes.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        self.axes.axis('off')
        self.draw()

class CornerDetectionApp(QWidget):
    def __init__(self):
        super().__init__()
        self.first_image = None
        self.second_image = None
        self.first_image_gray = None
        self.second_image_gray = None
        
        self.harris_corners1 = None
        self.shi_tomasi_corners1 = None
        self.sift_keypoints1 = None
        self.sift_descriptors1 = None
        self.sift_keypoints2 = None
        self.sift_descriptors2 = None
        self.matches = None

        self.dog_pyramid = None
        self.gradient_orientations = None
        
        self.initUI()
        self.connectSignalsSlots()
        
    def _create_detector_tab_ui(self, tab_name: str):
        """Creates the UI content for detector tabs (Harris, Shi-Tomasi, SIFT) with input and output image containers."""
        detector_tab_page_widget = QWidget()
        detector_tab_page_layout = QVBoxLayout(detector_tab_page_widget)
        detector_tab_page_layout.setSpacing(15)
        detector_tab_page_layout.setContentsMargins(10, 10, 10, 10)
    
        image_frame = QFrame()
        image_frame.setObjectName("imageFrame")
        image_frame.setMinimumSize(400, 300)
        image_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img_layout = QHBoxLayout(image_frame)  
    
        input_layout = QVBoxLayout()
        img_label = QLabel(f"Input Image ({tab_name})")
        img_label.setAlignment(Qt.AlignCenter)
        input_layout.addWidget(img_label)
        input_canvas = MplCanvas(image_frame)
        input_layout.addWidget(input_canvas)
        img_layout.addLayout(input_layout)
    
        output_layout = QVBoxLayout()
        output_label = QLabel(f"Output ({tab_name})")
        output_label.setAlignment(Qt.AlignCenter)
        output_layout.addWidget(output_label)
        output_canvas = MplCanvas(image_frame)
        output_layout.addWidget(output_canvas)
        img_layout.addLayout(output_layout)
    
        detector_tab_page_layout.addWidget(image_frame)
        detector_tab_page_layout.setStretchFactor(image_frame, 1)
    
        sliders_layout = QVBoxLayout()
        sliders_layout.setSpacing(10)
    
        param_layout = QHBoxLayout()
        param_label_text = "Parameter:" 
        if tab_name == "Harris":
            param_label_text = "Harris K:"
        elif tab_name == "Shi-Tomasi":
            param_label_text = "Max Corners:"
        elif tab_name == "SIFT":
            param_label_text = "Num Features:"
    
        param_label = QLabel(param_label_text)
        param_label.setFixedWidth(120)
        param_slider = QSlider(Qt.Horizontal)
    
        if tab_name == "Harris":
            param_slider.setMinimum(1)
            param_slider.setMaximum(100)
            param_slider.setValue(20)
            param_value_label = QLabel("0.20")
            param_slider.valueChanged.connect(
                lambda val, label=param_value_label:
                label.setText(f"{val/100:.2f}")
            )
        elif tab_name == "Shi-Tomasi":
            param_slider.setMinimum(10)
            param_slider.setMaximum(200)
            param_slider.setValue(50)
            param_value_label = QLabel("50")
            param_slider.valueChanged.connect(
                lambda val, label=param_value_label: 
                label.setText(str(val))
            )
        elif tab_name == "SIFT":
            param_slider.setMinimum(10)
            param_slider.setMaximum(500)
            param_slider.setValue(100)
            param_value_label = QLabel("100")
            param_slider.valueChanged.connect(
                lambda val, label=param_value_label: 
                label.setText(str(val))
            )
    
        param_value_label.setFixedWidth(30)
        param_layout.addWidget(param_label)
        param_layout.addWidget(param_slider)
        param_layout.addWidget(param_value_label)
        sliders_layout.addLayout(param_layout)
    
        threshold_layout = QHBoxLayout()
        threshold_label_text = "Threshold:"
        if tab_name == "Harris":
            threshold_label_text = "Threshold:"
        elif tab_name == "Shi-Tomasi":
            threshold_label_text = "Threshold:"
        elif tab_name == "SIFT":
            threshold_label_text = "Contrast Thresh.:"
    
        threshold_label = QLabel(threshold_label_text)
        threshold_label.setFixedWidth(120)
        threshold_slider = QSlider(Qt.Horizontal)
    
        if tab_name == "Harris":
            threshold_slider.setMinimum(1)
            threshold_slider.setMaximum(300) 
            threshold_slider.setValue(20)
            threshold_value_label = QLabel("0.020")
            threshold_slider.valueChanged.connect(
                lambda val, label=threshold_value_label:
                label.setText(f"{val/1000:.3f}")
            )
        elif tab_name == "Shi-Tomasi":
            threshold_slider.setMinimum(1)
            threshold_slider.setMaximum(50)
            threshold_slider.setValue(10)
            threshold_value_label = QLabel("0.1")
            threshold_slider.valueChanged.connect(
                lambda val, label=threshold_value_label: 
                label.setText(f"{val/100:.2f}")
            )
        elif tab_name == "SIFT":
            threshold_slider.setMinimum(1)
            threshold_slider.setMaximum(50)
            threshold_slider.setValue(10)
            threshold_value_label = QLabel("0.1")
            threshold_slider.valueChanged.connect(
                lambda val, label=threshold_value_label: 
                label.setText(f"{val/100:.2f}")
            )
    
        threshold_value_label.setFixedWidth(30)
        threshold_layout.addWidget(threshold_label)
        threshold_layout.addWidget(threshold_slider)
        threshold_layout.addWidget(threshold_value_label)
        sliders_layout.addLayout(threshold_layout)
    
        run_button = QPushButton(f"Run {tab_name} Detection")
        sliders_layout.addWidget(run_button)
    
        detector_tab_page_layout.addLayout(sliders_layout)
    
        if tab_name == "Harris":
            self.harris_param_slider = param_slider
            self.harris_threshold_slider = threshold_slider
            self.harris_run_button = run_button
            self.harris_canvas = input_canvas
            self.harris_output_canvas = output_canvas
        elif tab_name == "Shi-Tomasi":
            self.shi_tomasi_param_slider = param_slider
            self.shi_tomasi_threshold_slider = threshold_slider
            self.shi_tomasi_run_button = run_button
            self.shi_tomasi_canvas = input_canvas
            self.shi_tomasi_output_canvas = output_canvas
        elif tab_name == "SIFT":
            self.sift_param_slider = param_slider
            self.sift_threshold_slider = threshold_slider
            self.sift_run_button = run_button
            self.sift_canvas = input_canvas
            self.sift_output_canvas = output_canvas
    
        return detector_tab_page_widget

    def _create_matching_tab_ui(self):
        """Creates the UI content for the Matching tab."""
        matching_tab_page_widget = QWidget()
        matching_tab_page_layout = QVBoxLayout(matching_tab_page_widget)
        matching_tab_page_layout.setSpacing(15)
        matching_tab_page_layout.setContentsMargins(10, 10, 10, 10)

        image_area_layout = QHBoxLayout()
        image_area_layout.setSpacing(20)

        image1_frame = QFrame()
        image1_frame.setObjectName("imageFrame")
        image1_frame.setMinimumSize(200, 200)
        image1_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img1_layout = QVBoxLayout(image1_frame)
        img1_label = QLabel("First Image")
        img1_label.setAlignment(Qt.AlignCenter)
        img1_layout.addWidget(img1_label)
        
        self.matching_img1_canvas = MplCanvas(image1_frame)
        img1_layout.addWidget(self.matching_img1_canvas)
        image_area_layout.addWidget(image1_frame)

        image2_frame = QFrame()
        image2_frame.setObjectName("imageFrame")
        image2_frame.setMinimumSize(200, 200)
        image2_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        img2_layout = QVBoxLayout(image2_frame)
        img2_label = QLabel("Second Image")
        img2_label.setAlignment(Qt.AlignCenter)
        img2_layout.addWidget(img2_label)
        
        self.matching_img2_canvas = MplCanvas(image2_frame)
        img2_layout.addWidget(self.matching_img2_canvas)
        image_area_layout.addWidget(image2_frame)
        
        matching_tab_page_layout.addLayout(image_area_layout)
        matching_tab_page_layout.setStretchFactor(image_area_layout, 1)

        min_keypoint_layout = QHBoxLayout()
        min_keypoint_label = QLabel("Minimum Keypoint matches")
        self.min_keypoint_slider = QSlider(Qt.Horizontal)
        self.min_keypoint_slider.setMinimum(1)
        self.min_keypoint_slider.setMaximum(100)
        self.min_keypoint_slider.setValue(10)
        self.min_keypoint_value_label = QLabel("10")
        self.min_keypoint_value_label.setFixedWidth(30)
        self.min_keypoint_slider.valueChanged.connect(
            lambda val: self.min_keypoint_value_label.setText(str(val))
        )
        min_keypoint_layout.addWidget(min_keypoint_label)
        min_keypoint_layout.addWidget(self.min_keypoint_slider)
        min_keypoint_layout.addWidget(self.min_keypoint_value_label)
        matching_tab_page_layout.addLayout(min_keypoint_layout)

        self.btn_compare_images = QPushButton("Compare Images")
        matching_tab_page_layout.addWidget(self.btn_compare_images)

        bottom_buttons_layout = QHBoxLayout()
        bottom_buttons_layout.setSpacing(10)
        self.btn_show_dog = QPushButton("Show DoG")
        self.btn_show_keypoints = QPushButton("Show Keypoints")
        self.btn_show_gradient = QPushButton("Show Gradient Orientation")
        bottom_buttons_layout.addWidget(self.btn_show_dog)
        bottom_buttons_layout.addWidget(self.btn_show_keypoints)
        bottom_buttons_layout.addWidget(self.btn_show_gradient)
        matching_tab_page_layout.addLayout(bottom_buttons_layout)
        
        return matching_tab_page_widget
    
    def initUI(self):
        self.setWindowTitle('Corner Detection and Image Matching')
        self.setGeometry(100, 100, 1000, 650)

        self.setStyleSheet("""
            QWidget {
                background-color: #2E3440;
                color: #D8DEE9;
                font-size: 10pt;
            }
            QPushButton {
                background-color: #434C5E;
                border: 1px solid #4C566A;
                padding: 10px;
                border-radius: 5px;
                min-height: 30px;
            }
            QPushButton:hover {
                background-color: #4C566A;
            }
            QPushButton:pressed {
                background-color: #3B4252;
            }
            QLabel#mainTitle {
                padding: 5px;
                font-size: 18pt;
            }
            QLabel#leftPanelHeader {
                padding: 5px;
                font-size: 9pt;
            }
            QLabel {
                 padding: 5px;
                 font-size: 10pt;
            }
            QTabWidget::pane {
                border: 1px solid #4C566A;
            }
            QTabBar::tab {
                background: #3B4252;
                color: #ECEFF4;
                padding: 10px 15px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
                min-width: 100px;
                border: 1px solid #4C566A;
                border-bottom: none;
                margin-right: 2px;
            }
            QTabBar::tab:selected {
                background: #434C5E;
                border: 1px solid #88C0D0;
                border-bottom: 1px solid #434C5E;
                color: #ECEFF4;
            }
            QTabBar::tab:!selected:hover {
                background: #4C566A;
            }
            QFrame#imageFrame {
                background-color: #3B4252;
                border: 1px solid #4C566A;
                border-radius: 5px;
                color: #D8DEE9;
            }
            QSlider::groove:horizontal {
                border: 1px solid #4C566A;
                height: 8px;
                background: #434C5E;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #88C0D0;
                border: 1px solid #81A1C1;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
        """)

        main_layout = QHBoxLayout()
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(20)

        left_panel_layout = QVBoxLayout()
        left_panel_layout.setSpacing(10)
        left_panel_layout.setAlignment(Qt.AlignTop)
        load_images_label = QLabel("LOAD IMAGES")
        load_images_label.setObjectName("leftPanelHeader")
        load_images_label.setFont(QFont("Arial", 9, QFont.Bold))
        left_panel_layout.addWidget(load_images_label)
        self.btn_load_first = QPushButton("Load First Image")
        self.btn_load_second = QPushButton("Load Second Image")
        left_panel_layout.addWidget(self.btn_load_first)
        left_panel_layout.addWidget(self.btn_load_second)
        left_panel_layout.addSpacing(20)
        corner_detection_label = QLabel("CORNER DETECTION")
        corner_detection_label.setObjectName("leftPanelHeader")
        corner_detection_label.setFont(QFont("Arial", 9, QFont.Bold))
        left_panel_layout.addWidget(corner_detection_label)
        self.btn_harris = QPushButton("Harris Corner")
        self.btn_shi_tomasi = QPushButton("Shi-Tomasi")
        self.btn_sift = QPushButton("SIFT")
        left_panel_layout.addWidget(self.btn_harris)
        left_panel_layout.addWidget(self.btn_shi_tomasi)
        left_panel_layout.addWidget(self.btn_sift)
        left_panel_layout.addSpacing(20)
        evaluate_label = QLabel("EVALUATE PERFORMANCE")
        evaluate_label.setObjectName("leftPanelHeader")
        evaluate_label.setFont(QFont("Arial", 9, QFont.Bold))
        left_panel_layout.addWidget(evaluate_label)
        self.btn_evaluate = QPushButton("Evaluate Perform..")
        left_panel_layout.addWidget(self.btn_evaluate)
        left_panel_layout.addStretch(1)
        self.btn_info_help = QPushButton("info / Help")
        left_panel_layout.addWidget(self.btn_info_help)
        left_widget = QWidget()
        left_widget.setLayout(left_panel_layout)
        left_widget.setFixedWidth(200)
        main_layout.addWidget(left_widget)

        right_panel_layout = QVBoxLayout()
        right_panel_layout.setSpacing(15)

        title_label = QLabel("Corner Detection and Image Matching")
        title_label.setObjectName("mainTitle")
        title_label.setFont(QFont("Arial", 18, QFont.Bold))
        title_label.setAlignment(Qt.AlignLeft)
        right_panel_layout.addWidget(title_label)

        self.tabs = QTabWidget()
        
        harris_tab_content = self._create_detector_tab_ui("Harris")
        self.tabs.addTab(harris_tab_content, "Harris")

        shi_tomasi_tab_content = self._create_detector_tab_ui("Shi-Tomasi")
        self.tabs.addTab(shi_tomasi_tab_content, "Shi-Tomasi")

        sift_tab_content = self._create_detector_tab_ui("SIFT")
        self.tabs.addTab(sift_tab_content, "SIFT")

        matching_tab_content = self._create_matching_tab_ui()
        self.tabs.addTab(matching_tab_content, "Matching")
        
        right_panel_layout.addWidget(self.tabs)
        right_panel_layout.setStretchFactor(self.tabs, 1)

        main_layout.addLayout(right_panel_layout)
        main_layout.setStretchFactor(right_panel_layout, 1)

        self.setLayout(main_layout)
        
    def connectSignalsSlots(self):
        self.btn_load_first.clicked.connect(lambda: self.loadImage('first'))
        self.btn_load_second.clicked.connect(lambda: self.loadImage('second'))

        self.btn_harris.clicked.connect(lambda: self.tabs.setCurrentIndex(0))
        self.btn_shi_tomasi.clicked.connect(lambda: self.tabs.setCurrentIndex(1))
        self.btn_sift.clicked.connect(lambda: self.tabs.setCurrentIndex(2))

        self.harris_run_button.clicked.connect(self.runHarrisCornerDetection)
        self.harris_param_slider.sliderReleased.connect(self.runHarrisCornerDetection)
        self.harris_threshold_slider.sliderReleased.connect(self.runHarrisCornerDetection)

        self.shi_tomasi_run_button.clicked.connect(self.runShiTomasiCornerDetection)
        self.sift_run_button.clicked.connect(self.runSIFTDetection)

        self.btn_compare_images.clicked.connect(self.compareImages)
        self.btn_show_dog.clicked.connect(self.showDoG)
        self.btn_show_keypoints.clicked.connect(self.showKeypoints)
        self.btn_show_gradient.clicked.connect(self.showGradientOrientation)

        self.btn_info_help.clicked.connect(self.showHelp)
        self.btn_evaluate.clicked.connect(self.evaluatePerformance)
   
    def loadImage(self, which_image):
        options = QFileDialog.Options()
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Open Image File", "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*)",
            options=options
        )

        if file_path:
            try:
                img = cv2.imread(file_path)
                if img is None:
                    raise Exception("Failed to load image.")

                if which_image == 'first':
                    self.first_image = img
                    self.first_image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
                    self.harris_canvas.show_image(img)
                    self.shi_tomasi_canvas.show_image(img)
                    self.sift_canvas.show_image(img)
                    self.matching_img1_canvas.show_image(img)
                    
                    self.harris_output_canvas.clear()
                    self.shi_tomasi_output_canvas.clear()
                    self.sift_output_canvas.clear()
                else:  
                    self.second_image = img
                    self.second_image_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    self.matching_img2_canvas.show_image(img)

                QMessageBox.information(self, "Success", f"{which_image.capitalize()} image loaded successfully!")

            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to load image: {str(e)}")
    
    def checkFirstImageLoaded(self):
        if self.first_image is None:
            QMessageBox.warning(self, "Warning", "Please load the first image!")
            return False
        return True
    
    def checkImagesLoaded(self):
        if self.first_image is None or self.second_image is None:
            QMessageBox.warning(self, "Warning", "Please load both images for matching!")
            return False
        return True
    
    # ------------------------ Core Image Processing Functions -------------------- #
    
    def sobelOperator(self, image, direction='x'):
        if direction == 'x':
            kernel = np.array([[-1, 0, 1], 
                            [-2, 0, 2], 
                            [-1, 0, 1]])
        else:  
            kernel = np.array([[-1, -2, -1], 
                            [0, 0, 0], 
                            [1, 2, 1]])
        
        return cv2.filter2D(image.astype(float), -1, kernel)

    def gaussianKernel(self, size, sigma):
        """
        Generate a Gaussian kernel.
        
        Args:
            size: Kernel size (odd number)
            sigma: Standard deviation of the Gaussian
            
        Returns:
            2D Gaussian kernel
        """
        if size % 2 == 0:
            size += 1
        
        k = (size - 1) // 2
        x = np.linspace(-k, k, size)

        gauss_1d = np.exp(-0.5 * (x / sigma) ** 2)
        gauss_1d /= gauss_1d.sum()

        gauss_2d = np.outer(gauss_1d, gauss_1d)
        
        return gauss_2d
    
    def gaussianBlur(self, image, size, sigma):
        if size == 0:
            size = max(1, int(6 * sigma + 1))
            if size % 2 == 0: 
                size += 1
        
        return cv2.GaussianBlur(image, (size, size), sigma)
    
    # ------------------------ Harris Corner Detection ------------------------ #
    
    def runHarrisCornerDetection(self):
        if not self.checkFirstImageLoaded():
            return

        k = self.harris_param_slider.value() / 100
        threshold = self.harris_threshold_slider.value() / 1000  

        self.harris_corners1, img_corners = self.harrisCornerDetection(
            self.first_image, self.first_image_gray, k, threshold
        )

        print(f"[DEBUG] Harris: Detected {len(self.harris_corners1)} corners")
        self.harris_output_canvas.show_image(img_corners)

    def harrisCornerDetection(self, original_img, gray_img, k=0.04, threshold_ratio=0.01):
        """
        Harris Corner Detection with fixed raw R response and thresholding logic.

        Args:
            original_img: Original BGR image
            gray_img: Grayscale version
            k: Harris detector free parameter
            threshold_ratio: Percentage of max R to use as threshold (e.g., 0.01 for 1%)

        Returns:
            corners: List of (x, y) tuples for detected corners
            img_out: Image with drawn corners
        """
        gray = np.float32(gray_img)

        dx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        dy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        dx2 = dx * dx
        dy2 = dy * dy
        dxy = dx * dy

        dx2 = cv2.GaussianBlur(dx2, (3, 3), 1)
        dy2 = cv2.GaussianBlur(dy2, (3, 3), 1)
        dxy = cv2.GaussianBlur(dxy, (3, 3), 1)

        det = dx2 * dy2 - dxy ** 2
        trace = dx2 + dy2
        R = det - k * (trace ** 2)

        threshold = threshold_ratio * R.max()

        corners = []
        img_out = original_img.copy()
        height, width = R.shape

        for y in range(2, height - 2):
            for x in range(2, width - 2):
                if R[y, x] > threshold:
                    window = R[y - 2:y + 3, x - 2:x + 3]
                    if R[y, x] == np.max(window):
                        corners.append((x, y))

        print(f"Harris: Found {len(corners)} corners with k={k}, threshold={threshold:.5f}")

        for i, (x, y) in enumerate(corners):
            if i >= 300:
                break
            cv2.circle(img_out, (x, y), 6, (0, 255, 0), -1) 

        return corners, img_out

    # ------------------------ Shi-Tomasi Corner Detection ------------------ #
    
    def runShiTomasiCornerDetection(self):
        if not self.checkFirstImageLoaded():
            return

        max_corners = self.shi_tomasi_param_slider.value()
        quality_level = self.shi_tomasi_threshold_slider.value() / 100

        self.shi_tomasi_corners1, img_corners = self.shiTomasiCornerDetection(
            self.first_image, self.first_image_gray, max_corners, quality_level
        )

        self.shi_tomasi_output_canvas.show_image(img_corners)

        QMessageBox.information(self, "Success", "Shi-Tomasi corner detection completed!")
    
    def shiTomasiCornerDetection(self, original_img, gray_img, max_corners=50, quality_level=0.1, min_distance=10):
        corners = cv2.goodFeaturesToTrack(
            gray_img,
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance
        )
    
        result_img = original_img.copy()
        
        if corners is not None:
            corners = corners.reshape(-1, 2)
            for x, y in corners:
                x, y = int(x), int(y)
                cv2.circle(result_img, (x, y), 5, (0, 0, 255), 2)
        else:
            corners = []
        
        return [(int(x), int(y)) for x, y in corners], result_img
    
    # ------------------------ SIFT Detection ---------------------------- #
    
    def runSIFTDetection(self):
        """Run feature detection on the first image with fallback to ORB if SIFT fails."""
        if not self.checkFirstImageLoaded():
            return
    
        num_features = self.sift_param_slider.value()
        contrast_threshold = self.sift_threshold_slider.value() / 100
    
        e1 = None
        e2 = None
    
        try:
            sift = cv2.SIFT_create(
                nfeatures=num_features,
                contrastThreshold=contrast_threshold
            )
            self.sift_keypoints1, self.sift_descriptors1 = sift.detectAndCompute(self.first_image_gray, None)
            self._prepareSIFTVisualizations()
            img_keypoints = self.first_image.copy()
            cv2.drawKeypoints(self.first_image, self.sift_keypoints1, img_keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.sift_output_canvas.show_image(img_keypoints)
            QMessageBox.information(self, "Success", f"SIFT detection completed! Found {len(self.sift_keypoints1)} keypoints.")
            return
        except Exception as ex1:
            e1 = ex1
            print(f"Standard SIFT failed: {str(e1)}")

        try:
            sift = cv2.xfeatures2d.SIFT_create(
                nfeatures=num_features,
                contrastThreshold=contrast_threshold
            )
            self.sift_keypoints1, self.sift_descriptors1 = sift.detectAndCompute(self.first_image_gray, None)
            self._prepareSIFTVisualizations()
            img_keypoints = self.first_image.copy()
            cv2.drawKeypoints(self.first_image, self.sift_keypoints1, img_keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.sift_output_canvas.show_image(img_keypoints)
            QMessageBox.information(self, "Success", f"SIFT detection completed with legacy module! Found {len(self.sift_keypoints1)} keypoints.")
            return
        except Exception as ex2:
            e2 = ex2
            print(f"Legacy SIFT failed: {str(e2)}")
           
        try:
            QMessageBox.information(self, "Fallback", "Using ORB detector as SIFT is not available in your OpenCV version.")
            orb = cv2.ORB_create(nfeatures=num_features)
            self.sift_keypoints1, self.sift_descriptors1 = orb.detectAndCompute(self.first_image_gray, None)
            self._prepareSIFTVisualizations()
            img_keypoints = self.first_image.copy()
            cv2.drawKeypoints(self.first_image, self.sift_keypoints1, img_keypoints,
                        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            self.sift_output_canvas.show_image(img_keypoints)
            QMessageBox.information(self, "Success", f"ORB detection completed as fallback! Found {len(self.sift_keypoints1)} keypoints.")
        except Exception as e3:
            error_message = f"All feature detection methods failed:\n\n" \
                            f"SIFT error: {str(e1)}\n\n" \
                            f"Legacy SIFT error: {str(e2)}\n\n" \
                            f"ORB error: {str(e3)}"
            QMessageBox.critical(self, "Error", error_message)
    
    def _prepareSIFTVisualizations(self):
        dx = cv2.Sobel(self.first_image_gray.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        dy = cv2.Sobel(self.first_image_gray.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        self.gradient_orientations = np.arctan2(dy, dx) * 180 / np.pi

        self.dog_pyramid = []
        octave = []
        for i in range(4): 
            blurred1 = cv2.GaussianBlur(self.first_image_gray.astype(np.float32), (0, 0), sigmaX=1.0 * (1.5**i))
            blurred2 = cv2.GaussianBlur(self.first_image_gray.astype(np.float32), (0, 0), sigmaX=1.0 * (1.5**(i+1)))
            dog = blurred2 - blurred1
            octave.append(dog)
        self.dog_pyramid.append(octave)
    
    def siftDetection(self, original_img, gray_img, num_features=100, contrast_threshold=0.1):
        """
        Scale-Invariant Feature Transform (SIFT) implementation from scratch.
        
        Args:
            original_img: Original RGB image
            gray_img: Grayscale version of the image
            num_features: Maximum number of features to return
            contrast_threshold: Threshold for keypoint contrast
            
        Returns:
            keypoints: List of keypoint objects (x, y, scale, orientation)
            descriptors: Array of descriptors for each keypoint
            result_img: Image with keypoints marked
            dog_pyramid: Difference of Gaussian pyramid
            gradient_orientations: Gradient orientations
        """
        num_octaves = 4
        scales_per_octave = 5
        sigma_init = 1.6
        
        gaussian_pyramid, dog_pyramid = self.buildDoGPyramid(
            gray_img, num_octaves, scales_per_octave, sigma_init
        )
        
        keypoint_candidates = self.detectDoGKeypoints(
            dog_pyramid, num_octaves, scales_per_octave, contrast_threshold
        )
        
        gradient_mags, gradient_orientations = self.computeGradientMagnitudesOrientations(gray_img)
        
        keypoints = self.assignOrientations(
            keypoint_candidates, gaussian_pyramid, gradient_mags, gradient_orientations
        )
        
        descriptors = self.generateSIFTDescriptors(
            keypoints, gaussian_pyramid, gradient_mags, gradient_orientations
        )

        if len(keypoints) > num_features:
            keypoints_with_response = [(kp, np.abs(kp.response)) for kp in keypoints]
            keypoints_with_response.sort(key=lambda x: x[1], reverse=True)
            keypoints = [kp for kp, _ in keypoints_with_response[:num_features]]
            descriptors = descriptors[:num_features]
        
        result_img = original_img.copy()
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size * 1.2)
            angle = kp.angle

            cv2.circle(result_img, (x, y), size, (0, 255, 0), 1)
   
            end_x = int(x + size * np.cos(np.radians(angle)))
            end_y = int(y + size * np.sin(np.radians(angle)))
            cv2.line(result_img, (x, y), (end_x, end_y), (0, 255, 0), 1)
        
        return keypoints, descriptors, result_img, dog_pyramid, gradient_orientations
    
    def buildDoGPyramid(self, image, num_octaves, scales_per_octave, sigma_init):
        """
        Build a Difference of Gaussian pyramid from the input image.
        """
        image_float = image.astype(float)

        base_image = self.gaussianBlur(image_float, 0, sigma_init)
  
        octave_sizes = []
        base_height, base_width = base_image.shape
        
        for i in range(num_octaves):
            height = base_height // (2**i)
            width = base_width // (2**i)
            octave_sizes.append((height, width))
 
        k = 2 ** (1.0 / scales_per_octave)
        sigma_values = [sigma_init * (k**i) for i in range(scales_per_octave+3)]
        
        gaussian_pyramid = [[] for _ in range(num_octaves)]
        dog_pyramid = [[] for _ in range(num_octaves)]

        for octave in range(num_octaves):
            if octave == 0:
                current_image = base_image
            else:
                prev_octave_base = gaussian_pyramid[octave-1][0]
                current_image = cv2.resize(prev_octave_base, 
                                          (octave_sizes[octave][1], octave_sizes[octave][0]),
                                          interpolation=cv2.INTER_LINEAR)

            for scale, sigma in enumerate(sigma_values):
                if scale == 0:
                    gaussian_pyramid[octave].append(current_image)
                else:
                    sigma_diff = np.sqrt(sigma**2 - sigma_values[scale-1]**2)
                    blurred_image = self.gaussianBlur(gaussian_pyramid[octave][scale-1], 0, sigma_diff)
                    gaussian_pyramid[octave].append(blurred_image)
                
                if scale > 0:
                    dog = gaussian_pyramid[octave][scale] - gaussian_pyramid[octave][scale-1]
                    dog_pyramid[octave].append(dog)
        
        return gaussian_pyramid, dog_pyramid
    
    def detectDoGKeypoints(self, dog_pyramid, num_octaves, scales_per_octave, contrast_threshold):
        keypoints = []
        abs_threshold = contrast_threshold * 0.04
        
        for octave in range(num_octaves):
            for scale in range(1, len(dog_pyramid[octave])-1):
                prev_dog = dog_pyramid[octave][scale-1]
                curr_dog = dog_pyramid[octave][scale]
                next_dog = dog_pyramid[octave][scale+1]
                
                height, width = curr_dog.shape
 
                for y in range(1, height-1):
                    for x in range(1, width-1):
                        val = curr_dog[y, x]

                        if abs(val) < abs_threshold:
                            continue

                        local_extrema = False
                        is_max = True
                        is_min = True
                        
                        for dy in range(-1, 2):
                            for dx in range(-1, 2):
                                if prev_dog[y+dy, x+dx] > val:
                                    is_max = False
                                if prev_dog[y+dy, x+dx] < val:
                                    is_min = False
                                if dx != 0 or dy != 0:
                                    if curr_dog[y+dy, x+dx] > val:
                                        is_max = False
                                    if curr_dog[y+dy, x+dx] < val:
                                        is_min = False
                                if next_dog[y+dy, x+dx] > val:
                                    is_max = False
                                if next_dog[y+dy, x+dx] < val:
                                    is_min = False
                        
                        local_extrema = is_max or is_min
                        
                        if local_extrema:
                            kp_x = x * (2**octave)
                            kp_y = y * (2**octave)
                            kp_size = 3 * (2**octave)  
                            
                            kp = cv2.KeyPoint()
                            kp.pt = (kp_x, kp_y)
                            kp.size = kp_size
                            kp.octave = octave + (scale << 8)
                            kp.response = val  
                            keypoints.append(kp)
        
        return keypoints
    
    def computeGradientMagnitudesOrientations(self, image):
        dx = self.sobelOperator(image, direction='x')
        dy = self.sobelOperator(image, direction='y')

        magnitude = np.sqrt(dx**2 + dy**2)
        orientation = np.arctan2(dy, dx) * 180 / np.pi  
        
        return magnitude, orientation
    
    def assignOrientations(self, keypoints, gaussian_pyramid, gradient_mags, gradient_orientations):
        oriented_keypoints = []
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            octave = kp.octave & 255
            scale = (kp.octave >> 8) & 255

            scale_factor = 2**octave

            x_pyr = int(x / scale_factor)
            y_pyr = int(y / scale_factor)
   
            window_radius = int(kp.size / (2 * scale_factor))
            window_radius = max(1, window_radius)
            
            if octave < len(gaussian_pyramid) and scale < len(gaussian_pyramid[octave]):
                img_height, img_width = gradient_orientations.shape
                
                orientations_histogram = [0] * 36  
                
                for dy in range(-window_radius, window_radius + 1):
                    for dx in range(-window_radius, window_radius + 1):
                        if 0 <= y + dy < img_height and 0 <= x + dx < img_width:
                            
                            mag = gradient_mags[y + dy, x + dx]
                            ori = gradient_orientations[y + dy, x + dx]
                            
                            if ori < 0:
                                ori += 360

                            bin_idx = int(ori / 10) % 36
                            dist_sq = dx**2 + dy**2
                            weight = mag * np.exp(-dist_sq / (2 * window_radius**2))
                            
                            orientations_histogram[bin_idx] += weight
                
                smoothed_histogram = orientations_histogram.copy()
                for i in range(36):
                    prev_idx = (i - 1) % 36
                    next_idx = (i + 1) % 36
                    smoothed_histogram[i] = 0.25 * orientations_histogram[prev_idx] + \
                                           0.5 * orientations_histogram[i] + \
                                           0.25 * orientations_histogram[next_idx]
                
                max_bin = np.argmax(smoothed_histogram)
                max_val = smoothed_histogram[max_bin]
                
                kp.angle = max_bin * 10
                oriented_keypoints.append(kp)
                
                for i in range(36):
                    if i != max_bin and smoothed_histogram[i] > 0.8 * max_val:
                        new_kp = cv2.KeyPoint(kp.pt[0], kp.pt[1], kp.size, i * 10, kp.response, kp.octave)
                        oriented_keypoints.append(new_kp)
        
        return oriented_keypoints
    
    def generateSIFTDescriptors(self, keypoints, gaussian_pyramid, gradient_mags, gradient_orientations):
        descriptors = []
        
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            angle = kp.angle
            octave = kp.octave & 255
            scale = (kp.octave >> 8) & 255
            
            scale_factor = 2**octave

            angle_rad = np.radians(angle)
 
            cos_angle = np.cos(angle_rad)
            sin_angle = np.sin(angle_rad)

            pattern_size = 16 

            descriptor = np.zeros((4, 4, 8)) 

            region_size = kp.size / 4
            
            img_height, img_width = gradient_orientations.shape
            
            for i in range(4):
                for j in range(4):
                    region_center_x = (j - 1.5) * region_size
                    region_center_y = (i - 1.5) * region_size

                    for dy in range(-int(region_size/2), int(region_size/2) + 1):
                        for dx in range(-int(region_size/2), int(region_size/2) + 1):
                            rot_dx = dx * cos_angle - dy * sin_angle
                            rot_dy = dx * sin_angle + dy * cos_angle
           
                            sample_x = x + rot_dx
                            sample_y = y + rot_dy

                            if 0 <= int(sample_y) < img_height and 0 <= int(sample_x) < img_width:
                                mag = gradient_mags[int(sample_y), int(sample_x)]
                                ori = gradient_orientations[int(sample_y), int(sample_x)]
                                
                                rot_ori = ori - angle
                                if rot_ori < 0:
                                    rot_ori += 360
                                
                                bin_idx = int(rot_ori / 45) % 8
                                
                                dist_x = (rot_dx - region_center_x) / region_size
                                dist_y = (rot_dy - region_center_y) / region_size
                                weight = np.exp(-(dist_x**2 + dist_y**2) / 8)
                                
                                descriptor[i, j, bin_idx] += mag * weight
            
            flat_descriptor = descriptor.flatten()
            
            norm = np.linalg.norm(flat_descriptor)
            if norm > 0:
                flat_descriptor /= norm
            
            flat_descriptor = np.clip(flat_descriptor, 0, 0.2)
            norm = np.linalg.norm(flat_descriptor)
            if norm > 0:
                flat_descriptor /= norm
            
            descriptors.append(flat_descriptor)
        
        return np.array(descriptors)
    
    # ------------------------ Image Matching ---------------------------- #
    
    def compareImages(self):
        if not self.checkImagesLoaded():
            return
        
        if self.sift_keypoints1 is None:
            QMessageBox.warning(self, "Warning", "Please run SIFT detection on the first image first!")
            self.runSIFTDetection()
            
        if self.sift_keypoints2 is None:
            try:
                num_features = self.sift_param_slider.value()
                contrast_threshold = self.sift_threshold_slider.value() / 100
                
                sift = cv2.SIFT_create(
                    nfeatures=num_features,
                    contrastThreshold=contrast_threshold
                )
                
                self.sift_keypoints2, self.sift_descriptors2 = sift.detectAndCompute(self.second_image_gray, None)
                
            except Exception as e:
                QMessageBox.critical(self, "Error", f"SIFT detection failed on second image: {str(e)}")
                return
        
        try:
            min_matches = self.min_keypoint_slider.value()
            
            bf = cv2.BFMatcher()
            matches = bf.knnMatch(self.sift_descriptors1, self.sift_descriptors2, k=2)
            
            good_matches = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
            
            good_matches = sorted(good_matches, key=lambda x: x.distance)
            
            self.matches = good_matches[:min_matches]
            
            plt.close('all')
            
            plt.figure(figsize=(12, 8), num="Image Matching Results")
            
            img_matches = cv2.drawMatches(
                self.first_image, self.sift_keypoints1,
                self.second_image, self.sift_keypoints2,
                self.matches, None,
                flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
            )
            
            plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
            plt.title("Image Matching Results", fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            plt.show()
            
            self.matching_img1_canvas.show_image(self.first_image)
            self.matching_img2_canvas.show_image(self.second_image)
            
            QMessageBox.information(self, "Success", f"Found {len(self.matches)} matching keypoints!")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Image matching failed: {str(e)}")
    
    def displayMatchesInNewWindow(self, img1, img2, keypoints1, keypoints2, matches):
        """
        Display matched keypoints between two images with green lines in a new window.
        
        Args:
            img1: First image
            img2: Second image
            keypoints1: List of keypoints from first image
            keypoints2: List of keypoints from second image
            matches: List of DMatch objects
        """
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        output_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        
        output_img[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        output_img[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        for match in matches:
            kp1 = keypoints1[match.queryIdx]
            kp2 = keypoints2[match.trainIdx]
            
            x1, y1 = map(int, kp1.pt)
            x2, y2 = map(int, kp2.pt)
            x2 += w1
            
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            cv2.circle(output_img, (x1, y1), 4, (0, 255, 0), 1)
            cv2.circle(output_img, (x2, y2), 4, (0, 255, 0), 1)
        
        plt.figure(figsize=(12, 8))
        plt.title("Image Matching Results")
        plt.imshow(output_img)
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def matchKeypoints(self, keypoints1, keypoints2, descriptors1, descriptors2, min_matches=10):
        """
        Match keypoints between two images using their descriptors.
        
        Args:
            keypoints1: List of keypoints from first image
            keypoints2: List of keypoints from second image
            descriptors1: Descriptor array for first image keypoints
            descriptors2: Descriptor array for second image keypoints
            min_matches: Minimum number of matches to return
            
        Returns:
            matches: List of DMatch objects
        """
        matches = []
        
        for i, desc1 in enumerate(descriptors1):
            distances = []
            for j, desc2 in enumerate(descriptors2):
                dist = np.linalg.norm(desc1 - desc2)
                distances.append((j, dist))
            
            distances.sort(key=lambda x: x[1])
            
            if len(distances) >= 2:
                best_idx, best_dist = distances[0]
                second_best_idx, second_best_dist = distances[1]
                
                if best_dist < 0.75 * second_best_dist:
                    match = cv2.DMatch()
                    match.queryIdx = i
                    match.trainIdx = best_idx
                    match.distance = best_dist
                    matches.append(match)
        
        matches.sort(key=lambda x: x.distance)
        
        return matches[:max(min_matches, len(matches))]
    
    def displayMatches(self, img1, img2, keypoints1, keypoints2, matches):
        """
        Display matched keypoints between two images with green lines.
        
        Args:
            img1: First image
            img2: Second image
            keypoints1: List of keypoints from first image
            keypoints2: List of keypoints from second image
            matches: List of DMatch objects
        """
        self.matching_canvas.clear()
        
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        
        output_img = np.zeros((max(h1, h2), w1 + w2, 3), dtype=np.uint8)
        
        output_img[:h1, :w1] = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
        output_img[:h2, w1:w1+w2] = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        for match in matches:
            kp1 = keypoints1[match.queryIdx]
            kp2 = keypoints2[match.trainIdx]
            
            x1, y1 = map(int, kp1.pt)
            x2, y2 = map(int, kp2.pt)
            
            x2 += w1
            
            cv2.line(output_img, (x1, y1), (x2, y2), (0, 255, 0), 1)
            
            cv2.circle(output_img, (x1, y1), 4, (0, 255, 0), 1)
            cv2.circle(output_img, (x2, y2), 4, (0, 255, 0), 1)
        
        self.matching_canvas.axes.imshow(output_img)
        self.matching_canvas.axes.axis('off')
        self.matching_canvas.draw()
        
    # ------------------------ SIFT Visualization ---------------------------- #
        
    def showDoG(self):
        """Show Difference of Gaussians (DoG) for the first image."""
        if self.dog_pyramid is None:
            QMessageBox.warning(self, "Warning", "SIFT detection must be run first!")
            return
        
        figure = plt.figure(figsize=(10, 8))
        plt.suptitle("Difference of Gaussian (DoG) Pyramid", fontsize=16)
        
        num_octaves = len(self.dog_pyramid)
        scales_per_octave = len(self.dog_pyramid[0])
        
        for octave in range(min(num_octaves, 4)):
            for scale in range(min(scales_per_octave, 4)):
                idx = octave * min(scales_per_octave, 4) + scale + 1
                ax = figure.add_subplot(min(num_octaves, 4), min(scales_per_octave, 4), idx)
                
                dog_img = self.dog_pyramid[octave][scale]
                dog_img_norm = (dog_img - dog_img.min()) / (dog_img.max() - dog_img.min())
                
                ax.imshow(dog_img_norm, cmap='gray')
                ax.set_title(f"Oct {octave}, Scale {scale}")
                ax.axis('off')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def showKeypoints(self):
        """Show keypoints detected by SIFT."""
        if self.sift_keypoints1 is None:
            QMessageBox.warning(self, "Warning", "SIFT detection must be run first!")
            return
        
        figure = plt.figure(figsize=(10, 8))
        plt.suptitle("SIFT Keypoints", fontsize=16)
        
        result_img = self.first_image.copy()
        
        # Draw all keypoints with their scale and orientation
        for kp in self.sift_keypoints1:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            size = int(kp.size)
            angle = kp.angle
            octave = kp.octave & 255
            
            # Draw keypoint as a circle
            cv2.circle(result_img, (x, y), size, (0, 255, 0), 1)
            
            # Draw orientation line
            end_x = int(x + size * np.cos(np.radians(angle)))
            end_y = int(y + size * np.sin(np.radians(angle)))
            cv2.line(result_img, (x, y), (end_x, end_y), (0, 255, 0), 1)
        
        # Display the image
        plt.imshow(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.tight_layout()
        plt.show()
    
    def showGradientOrientation(self):
        """Show gradient orientation of the first image."""
        if self.gradient_orientations is None:
            QMessageBox.warning(self, "Warning", "SIFT detection must be run first!")
            return
        
        # Create a figure to display gradient orientations
        figure = plt.figure(figsize=(12, 8))
        plt.suptitle("Gradient Orientation", fontsize=16)
        
        # Calculate gradient magnitude
        dx = self.sobelOperator(self.first_image_gray, direction='x')
        dy = self.sobelOperator(self.first_image_gray, direction='y')
        magnitude = np.sqrt(dx**2 + dy**2)
        
        # Normalize for better visibility
        mag_norm = np.clip(magnitude / magnitude.max(), 0, 1)
        
        # Convert orientations from degrees to hue (0-360 degrees to 0-1 range)
        orientation_hue = (self.gradient_orientations + 180) / 360.0
        
        # Create HSV image (hue = orientation, saturation = 1, value = magnitude)
        hsv = np.zeros((self.first_image_gray.shape[0], self.first_image_gray.shape[1], 3), dtype=np.float32)
        hsv[:, :, 0] = orientation_hue  # Hue
        hsv[:, :, 1] = 1.0              # Saturation
        hsv[:, :, 2] = mag_norm         # Value
        
        # Convert HSV to RGB
        rgb = cv2.cvtColor((hsv * 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        # Display the original image and gradient orientation
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(self.first_image, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.imshow(rgb)
        plt.title("Gradient Orientation")
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    # ------------------------ Helper Methods ---------------------------- #
    
    def showHelp(self):
        """Show help information."""
        help_text = """
        <b>Corner Detection and Image Matching</b>
        
        <p>This application demonstrates various corner detection algorithms and 
        feature matching using SIFT.</p>
        
        <p><b>How to use:</b></p>
        <ol>
            <li>Load an image using the 'Load First Image' button</li>
            <li>For matching, load a second image using 'Load Second Image'</li>
            <li>Choose a detection algorithm tab:
                <ul>
                    <li><b>Harris:</b> Adjust K parameter and threshold</li>
                    <li><b>Shi-Tomasi:</b> Set max corners and quality level</li>
                    <li><b>SIFT:</b> Adjust features count and contrast threshold</li>
                </ul>
            </li>
            <li>Run the detection using the corresponding 'Run' button</li>
            <li>For matching, go to 'Matching' tab and click 'Compare Images'</li>
        </ol>
        
        <p><b>Additional features:</b></p>
        <ul>
            <li>In the Matching tab, use 'Show DoG' to see the Difference of Gaussian pyramid</li>
            <li>Use 'Show Keypoints' to see all detected SIFT keypoints</li>
            <li>Use 'Show Gradient Orientation' to visualize gradient directions</li>
        </ul>
        """
        
        QMessageBox.information(self, "Help", help_text)
    
    def evaluatePerformance(self):
        """Evaluate the performance of corner detectors."""
        if not self.checkFirstImageLoaded():
            return

        plt.close('all')
        plt.figure(figsize=(18, 6))
        plt.suptitle("Corner Detection Evaluation", fontsize=16)

        # Harris
        harris_params = [0.02, 0.04, 0.06, 0.08]
        harris_corners_count = []
        for k in harris_params:
            corners, _ = self.harrisCornerDetection(self.first_image, self.first_image_gray, k=k, threshold_ratio=0.1)
            harris_corners_count.append(len(corners))

        # Shi-Tomasi
        shi_tomasi_qualities = [0.05, 0.1, 0.15, 0.2]
        shi_tomasi_corners_count = []
        for q in shi_tomasi_qualities:
            corners, _ = self.shiTomasiCornerDetection(self.first_image, self.first_image_gray, max_corners=200, quality_level=q)
            shi_tomasi_corners_count.append(len(corners))

        # SIFT
        sift_contrasts = [0.01, 0.03, 0.05, 0.07]
        sift_keypoints_count = []
        for c in sift_contrasts:
            try:
                sift = cv2.SIFT_create(nfeatures=200, contrastThreshold=c)
                keypoints, _ = sift.detectAndCompute(self.first_image_gray, None)
                sift_keypoints_count.append(len(keypoints))
            except Exception:
                sift_keypoints_count.append(0)

        # Plot Harris
        plt.subplot(1, 3, 1)
        plt.plot(harris_params, harris_corners_count, 'o-', color='green')
        plt.xlabel('Harris k parameter')
        plt.ylabel('Number of corners')
        plt.title('Harris Detector')
        plt.grid(True)

        # Plot Shi-Tomasi
        plt.subplot(1, 3, 2)
        plt.plot(shi_tomasi_qualities, shi_tomasi_corners_count, 'o-', color='blue')
        plt.xlabel('Quality level')
        plt.ylabel('Number of corners')
        plt.title('Shi-Tomasi Detector')
        plt.grid(True)

        # Plot SIFT
        plt.subplot(1, 3, 3)
        plt.plot(sift_contrasts, sift_keypoints_count, 'o-', color='orange')
        plt.xlabel('SIFT Contrast Threshold')
        plt.ylabel('Number of keypoints')
        plt.title('SIFT Detector')
        plt.grid(True)

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)
        
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = CornerDetectionApp()
    window.show()
    sys.exit(app.exec_())
