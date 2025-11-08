"""
PyQt5 GUI: 피스타치오 이미지 분류기
실행: python app.py
"""

import sys
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                              QPushButton, QLabel, QFileDialog, QMessageBox)
from PyQt5.QtGui import QPixmap, QFont
from PyQt5.QtCore import Qt
from PIL import Image
from tensorflow import keras


class PistachioClassifier(QMainWindow):
    """피스타치오 이미지 분류 GUI 애플리케이션"""

    def __init__(self):
        super().__init__()
        self.model = None
        self.class_names = ['Kirmizi', 'Siirt']
        self.init_ui()
        self.load_model()

    def init_ui(self):
        """UI 초기화"""
        # 윈도우 설정
        self.setWindowTitle('AI Image Classifier')
        self.setGeometry(100, 100, 600, 700)

        # 중앙 위젯 및 레이아웃
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout()
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # 제목
        title_label = QLabel('AI Image Classifier')
        title_font = QFont('Arial', 18, QFont.Bold)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(title_label)

        subtitle_label = QLabel('Deep Learning Term Project - Pistachio Classification')
        subtitle_font = QFont('Arial', 10)
        subtitle_label.setFont(subtitle_font)
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet('color: gray;')
        layout.addWidget(subtitle_label)

        layout.addSpacing(10)

        # Load Image 버튼
        self.load_btn = QPushButton('Load Image')
        self.load_btn.setFont(QFont('Arial', 12))
        self.load_btn.setMinimumHeight(40)
        self.load_btn.clicked.connect(self.load_image)
        self.load_btn.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 5px;
                padding: 10px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
        """)
        layout.addWidget(self.load_btn)

        layout.addSpacing(10)

        # 이미지 표시 영역
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setMinimumHeight(400)
        self.image_label.setMaximumHeight(400)
        self.image_label.setStyleSheet("""
            QLabel {
                border: 2px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
            }
        """)
        self.image_label.setText('No image loaded')
        self.image_label.setFont(QFont('Arial', 11))
        layout.addWidget(self.image_label)

        layout.addSpacing(10)

        # Classification Result 섹션
        result_title = QLabel('Classification Result')
        result_title.setFont(QFont('Arial', 14, QFont.Bold))
        layout.addWidget(result_title)

        # 구분선
        separator = QLabel('─' * 50)
        separator.setFont(QFont('Arial', 10))
        separator.setStyleSheet('color: #999;')
        layout.addWidget(separator)

        # 결과 텍스트 영역
        self.result_label = QLabel('')
        self.result_label.setFont(QFont('Arial', 12))
        self.result_label.setWordWrap(True)
        self.result_label.setStyleSheet('padding: 10px;')
        layout.addWidget(self.result_label)

        # 레이아웃 마무리
        layout.addStretch()
        central_widget.setLayout(layout)

        # 상태바
        self.statusBar().showMessage('Ready')

    def load_model(self):
        """학습된 모델 로딩"""
        model_path = 'pistachio_cnn_model.h5'

        if not os.path.exists(model_path):
            QMessageBox.critical(
                self,
                'Error',
                f'모델 파일을 찾을 수 없습니다: {model_path}\n\n'
                '먼저 train_pistachio_cnn.py를 실행하여 모델을 학습시켜주세요.'
            )
            self.statusBar().showMessage('Model not found')
            return

        try:
            self.model = keras.models.load_model(model_path)
            self.statusBar().showMessage('Model loaded successfully')
            QMessageBox.information(
                self,
                'Success',
                '모델이 성공적으로 로딩되었습니다!'
            )
        except Exception as e:
            QMessageBox.critical(
                self,
                'Error',
                f'모델 로딩 실패: {str(e)}'
            )
            self.statusBar().showMessage('Model loading failed')

    def load_image(self):
        """이미지 파일 선택 및 로딩"""
        if self.model is None:
            QMessageBox.warning(
                self,
                'Warning',
                '모델이 로드되지 않았습니다.'
            )
            return

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            'Select Image File',
            '',
            'Image Files (*.jpg *.jpeg *.png);;All Files (*)'
        )

        if not file_path:
            return

        try:
            self.statusBar().showMessage('Loading image...')
            image = Image.open(file_path)

            if image.mode != 'RGB':
                image = image.convert('RGB')

            self.display_image(image)

            self.statusBar().showMessage('Predicting...')
            self.predict_image(image)

            self.statusBar().showMessage(f'Image loaded: {os.path.basename(file_path)}')

        except Exception as e:
            QMessageBox.critical(
                self,
                'Error',
                f'이미지 처리 중 오류 발생:\n{str(e)}'
            )
            self.statusBar().showMessage('Error loading image')

    def display_image(self, pil_image):
        """이미지를 화면에 표시"""
        temp_path = 'temp_display_image.png'
        pil_image.save(temp_path)

        pixmap = QPixmap(temp_path)
        scaled_pixmap = pixmap.scaledToHeight(400, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

        try:
            os.remove(temp_path)
        except:
            pass

    def preprocess_image(self, pil_image):
        """이미지 전처리"""
        img_resized = pil_image.resize((120, 120))
        img_array = np.array(img_resized, dtype='float32')

        if len(img_array.shape) == 2:
            img_array = np.stack([img_array]*3, axis=-1)
        elif img_array.shape[2] == 4:
            img_array = img_array[:, :, :3]

        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        return img_array

    def predict_image(self, pil_image):
        """이미지 예측"""
        try:
            processed_image = self.preprocess_image(pil_image)
            predictions = self.model.predict(processed_image, verbose=0)[0]

            predicted_idx = np.argmax(predictions)
            predicted_class = self.class_names[predicted_idx]
            predicted_prob = predictions[predicted_idx]

            result_text = predicted_class
            self.result_label.setText(result_text)

            self.statusBar().showMessage(
                f'Prediction: {predicted_class} ({predicted_prob*100:.1f}%)'
            )

        except Exception as e:
            QMessageBox.critical(
                self,
                'Error',
                f'예측 중 오류 발생:\n{str(e)}'
            )
            self.statusBar().showMessage('Prediction failed')


def main():
    app = QApplication(sys.argv)

    # 애플리케이션 스타일 설정
    app.setStyle('Fusion')

    # 메인 윈도우 생성 및 표시
    window = PistachioClassifier()
    window.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()