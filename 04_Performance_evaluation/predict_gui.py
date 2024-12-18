import sys
import os
import pandas as pd
import joblib
import json
import warnings
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog, QTextEdit, QComboBox, QMessageBox
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

warnings.filterwarnings("ignore")

features_path = 'GA_output_ET.json'

def load_features(file_path):
    with open(file_path, 'r') as file:
        features = json.load(file)
    return features

features = load_features(features_path)

def get_model_paths(model_type):
    model_suffix = {
        'Random Forest': 'RF',
        'Support Vector Machine': 'SVM',
        'Naive Bayes': 'NB'
    }

    return {
        'SYN': f'./models/{model_suffix[model_type]}_SYN_1_model.pkl',
        'HTTP': f'./models/{model_suffix[model_type]}_HTTP_1_model.pkl',
        'ACK': f'./models/{model_suffix[model_type]}_ACK_1_model.pkl',
        'UDP': f'./models/{model_suffix[model_type]}_UDP_1_model.pkl',
        'ARP': f'./models/{model_suffix[model_type]}_ARP_1_model.pkl',
        'SP': f'./models/{model_suffix[model_type]}_SP_1_model.pkl',
        'BF': f'./models/{model_suffix[model_type]}_BF_1_model.pkl',
    }

def predict_attack(input_data, model_paths):
    results = []
    for attack, model_path in model_paths.items():
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        model = joblib.load(model_path)
        _features = features.get(attack, [])

        if not _features:
            raise ValueError(f"No features found for attack type: {attack}")

        try:
            filtered_data = input_data[_features[:-1]]
        except KeyError as e:
            raise ValueError(f"Missing required feature columns in the input data: {str(e)}")

        predictions = model.predict(filtered_data)

        results.extend([{"Attack_Type": attack, "Prediction": int(pred)} for pred in predictions])

    return results

def summarize_results(results, input_data):
    results_df = pd.DataFrame(results)
    attack_predictions = results_df[results_df["Prediction"] == 1]

    total_samples = len(input_data)
    summary = {}

    if not attack_predictions.empty:
        attack_counts = attack_predictions.groupby("Attack_Type").size()
        
        for attack_type, attack_count in attack_counts.items():
            non_attack_count = total_samples - attack_count
            summary[attack_type] = {
                "Detected_Attack_Count": attack_count,
                "Non_Attack_Count": non_attack_count
            }
    return summary


class AttackPredictionApp(QWidget):
    def __init__(self):
        super().__init__()

        # Thiết lập giao diện
        self.setWindowTitle("Network Attack Prediction")
        self.setGeometry(100, 100, 1000, 720)
        self.setStyleSheet("background-color: #f4f7f6; font-family: Arial, sans-serif;")

        # Layout chính
        layout = QVBoxLayout()

        # Tiêu đề ứng dụng
        self.title = QLabel("Network Attack Prediction")
        self.title.setStyleSheet("font-size: 28px; font-weight: bold; color: #4CAF50; text-align: center; padding-bottom: 20px;")
        layout.addWidget(self.title)

        # Layout phụ cho nút chọn tệp và hiển thị đường dẫn
        file_layout = QHBoxLayout()
        self.select_file_button = QPushButton("Select CSV File")
        self.select_file_button.setStyleSheet("background-color: #4CAF50; color: white; font-size: 16px; padding: 10px; border-radius: 5px;")
        self.select_file_button.clicked.connect(self.select_file)
        file_layout.addWidget(self.select_file_button)

        self.file_path_label = QLabel("No file selected")
        self.file_path_label.setStyleSheet("font-size: 14px; color: #555; margin-left: 10px;")
        file_layout.addWidget(self.file_path_label)

        layout.addLayout(file_layout)

        # Label cho dropdown model
        self.model_label = QLabel("Select Model")
        self.model_label.setStyleSheet("font-size: 18px; color: #333; margin-top: 20px;")
        layout.addWidget(self.model_label)

        # Dropdown chọn mô hình (hiển thị tên đầy đủ)
        self.model_combo = QComboBox()
        self.model_combo.addItems(["Random Forest", "Support Vector Machine", "Naive Bayes"])
        self.model_combo.setStyleSheet("font-size: 16px; padding: 10px;")
        self.model_combo.currentTextChanged.connect(self.update_model_label)  # Cập nhật tên mô hình khi thay đổi
        layout.addWidget(self.model_combo)

        # Label hiển thị tên mô hình đã chọn
        self.selected_model_label = QLabel("Selected Model: Random Forest")
        self.selected_model_label.setStyleSheet("font-size: 16px; color: #333; margin-top: 10px;")
        layout.addWidget(self.selected_model_label)

        # Nút bắt đầu dự đoán
        self.predict_button = QPushButton("Start Prediction")
        self.predict_button.setStyleSheet("background-color: #008CBA; color: white; font-size: 16px; padding: 10px; border-radius: 5px; margin-top: 20px;")
        self.predict_button.clicked.connect(self.start_prediction)
        self.predict_button.setEnabled(False)  # Disable until a file is selected
        layout.addWidget(self.predict_button)

        # Hiển thị kết quả dự đoán
        self.result_display = QTextEdit()
        self.result_display.setReadOnly(True)
        self.result_display.setStyleSheet("font-size: 14px; padding: 10px; background-color: #f9f9f9; border-radius: 5px;")
        layout.addWidget(self.result_display)

        self.setLayout(layout)

    def select_file(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            try:
                test_data = pd.read_csv(file_path)
                required_columns = list(features['SYN'])  # Dựa vào các cột cần thiết
                if not all(col in test_data.columns for col in required_columns):
                    raise ValueError("The selected CSV file does not contain the required columns.")

                self.file_path_label.setText(f"Selected File: {file_path}")
                self.predict_button.setEnabled(True)  # Enable prediction button
                self.selected_file = file_path

            except Exception as e:
                self.show_error("File Error", str(e))

    def start_prediction(self):
        self.result_display.clear()  # Xóa kết quả cũ

        # Lấy mô hình đã chọn
        model_type = self.model_combo.currentText()
        model_paths = get_model_paths(model_type)

        # Đọc dữ liệu từ CSV và thực hiện dự đoán
        try:
            test_data = pd.read_csv(self.selected_file)
            predict_results = predict_attack(test_data, model_paths)

            # Tổng hợp kết quả
            summarize = summarize_results(predict_results, test_data)

            if summarize:
                result_text = "Predict Attacks:\n"
                for attack_type, data in summarize.items():
                    result_text += (
                        f" - Attack Type: {attack_type}\n"
                        f"\tDetected: {data['Detected_Attack_Count']}, "
                        f"\tNon-Attack: {data['Non_Attack_Count']}\n"
                    )
            else:
                result_text = "No attacks detected."

            # Hiển thị kết quả sau khi dự đoán
            self.result_display.setText(result_text)
        except Exception as e:
            self.show_error("Prediction Error", f"An error occurred during prediction: {str(e)}")


    def update_model_label(self, model_type):
        self.selected_model_label.setText(f"Selected Model: {model_type}")

    def show_error(self, title, message):
        error_dialog = QMessageBox()
        error_dialog.setIcon(QMessageBox.Warning)
        error_dialog.setWindowTitle(title)
        error_dialog.setText(message)
        error_dialog.exec_()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = AttackPredictionApp()
    window.show()
    sys.exit(app.exec_())
