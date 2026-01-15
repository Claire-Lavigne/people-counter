import cv2
import numpy as np
import sys
import threading
import time
from PyQt5 import QtWidgets, QtGui, QtCore
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Load pre-trained model (YOLOv3-tiny for speed, or use another model as needed)
# Download weights and config from official YOLO sources if not present
YOLO_CONFIG = 'yolov3-tiny.cfg'  # Placeholder path
YOLO_WEIGHTS = 'yolov3-tiny.weights'  # Placeholder path
YOLO_CLASSES = 'coco.names'  # Placeholder path

net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
print("[INFO] Utilisation du CPU pour l'inférence.")
with open(YOLO_CLASSES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers().flatten()]



# --- PyQt5 GUI ---
class VideoThread(QtCore.QThread):
    change_pixmap_signal = QtCore.pyqtSignal(np.ndarray, int)
    def __init__(self, source):
        super().__init__()
        self.source = source
        self._run_flag = True
        self.history = []  # (timestamp, count)
    def run(self):
        cap = cv2.VideoCapture(self.source)
        while self._run_flag:
            ret, frame = cap.read()
            if not ret:
                break
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
            net.setInput(blob)
            outs = net.forward(output_layers)
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if classes[class_id] == 'person' and confidence > 0.5:
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            if isinstance(indexes, tuple):
                indexes = list(indexes)
            elif hasattr(indexes, 'flatten'):
                indexes = indexes.flatten().tolist()
            else:
                indexes = list(indexes)
            person_count = len(indexes)
            self.history.append((time.time(), person_count))
            for i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = (0, 255, 0)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.putText(frame, f'People Count: {person_count}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.putText(frame, 'No data is collected or transmitted.', (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            self.change_pixmap_signal.emit(frame, person_count)
            if not self._run_flag:
                break
        cap.release()
    def stop(self):
        self._run_flag = False
        self.wait()
    def get_history(self):
        return self.history

class MainWindow(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('People Counter')
        self.setGeometry(100, 100, 1200, 700)
        self.image_label = QtWidgets.QLabel(self)
        self.image_label.setGeometry(10, 50, 900, 640)
        self.combo = QtWidgets.QComboBox(self)
        self.combo.addItems([str(i) for i in range(5)])
        self.combo.setGeometry(10, 10, 100, 30)
        self.btn_start = QtWidgets.QPushButton('Démarrer', self)
        self.btn_start.setGeometry(120, 10, 100, 30)
        self.btn_stop = QtWidgets.QPushButton('Arrêter', self)
        self.btn_stop.setGeometry(230, 10, 100, 30)
        self.btn_stop.setEnabled(False)
        self.btn_start.clicked.connect(self.start_video)
        self.btn_stop.clicked.connect(self.stop_video)
        # Matplotlib Figure for real-time plot
        self.figure = Figure(figsize=(3, 4))
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setParent(self)
        self.canvas.setGeometry(930, 50, 250, 640)
        self.ax = self.figure.add_subplot(111)
        self.ax.set_title('Historique du comptage')
        self.ax.set_xlabel('Temps (s)')
        self.ax.set_ylabel('Personnes')
        self.plot_data = []
        self.plot_times = []
        self.thread = None
        self.start_time = None
        # Statistiques
        self.stats_label = QtWidgets.QLabel(self)
        self.stats_label.setGeometry(930, 10, 250, 30)
        self.stats_label.setText('Moyenne: -   Min: -   Max: -')
        # Bouton export PDF
        self.btn_export = QtWidgets.QPushButton('Exporter PDF', self)
        self.btn_export.setGeometry(1050, 10, 120, 30)
        self.btn_export.clicked.connect(self.export_pdf)
        def export_pdf(self):
            from reportlab.lib.pagesizes import A4
            from reportlab.pdfgen import canvas
            import tempfile
            import os
            # Sauvegarder le graphique comme image temporaire
            tmp_img = tempfile.mktemp(suffix='.png')
            self.figure.savefig(tmp_img)
            # Créer le PDF
            pdf_path, _ = QtWidgets.QFileDialog.getSaveFileName(self, 'Enregistrer le PDF', '', 'PDF Files (*.pdf)')
            if not pdf_path:
                os.remove(tmp_img)
                return
            c = canvas.Canvas(pdf_path, pagesize=A4)
            c.setFont('Helvetica', 14)
            c.drawString(50, 800, 'Rapport de comptage de personnes')
            c.setFont('Helvetica', 10)
            c.drawString(50, 780, f'Moyenne: {self.stats_label.text()}')
            c.drawImage(tmp_img, 50, 400, width=500, height=300)
            c.setFont('Helvetica', 10)
            c.drawString(50, 380, 'Historique (temps en secondes, personnes):')
            y = 360
            for t, v in zip(self.plot_times, self.plot_data):
                c.drawString(50, y, f'{t:.1f} s : {v}')
                y -= 12
                if y < 50:
                    c.showPage()
                    y = 800
            c.save()
            os.remove(tmp_img)
            QtWidgets.QMessageBox.information(self, 'Export PDF', 'Export PDF terminé !')
    def start_video(self):
        source = int(self.combo.currentText())
        self.thread = VideoThread(source)
        self.thread.change_pixmap_signal.connect(self.update_image)
        self.thread.start()
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.start_time = time.time()
        self.plot_data = []
        self.plot_times = []
        self.ax.clear()
        self.ax.set_title('Historique du comptage')
        self.ax.set_xlabel('Temps (s)')
        self.ax.set_ylabel('Personnes')
        self.canvas.draw()
    def stop_video(self):
        if self.thread:
            self.thread.stop()
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
    def update_image(self, cv_img, person_count):
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
        # Update plot
        if self.start_time is not None:
            t = time.time() - self.start_time
            self.plot_times.append(t)
            self.plot_data.append(person_count)
            self.ax.clear()
            self.ax.plot(self.plot_times, self.plot_data, color='blue')
            self.ax.set_title('Historique du comptage')
            self.ax.set_xlabel('Temps (s)')
            self.ax.set_ylabel('Personnes')
            self.canvas.draw()
            # Statistiques
            if self.plot_data:
                moyenne = sum(self.plot_data) / len(self.plot_data)
                minimum = min(self.plot_data)
                maximum = max(self.plot_data)
                self.stats_label.setText(f'Moyenne: {moyenne:.2f}   Min: {minimum}   Max: {maximum}')
            else:
                self.stats_label.setText('Moyenne: -   Min: -   Max: -')
    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(900, 640, QtCore.Qt.KeepAspectRatio)
        return QtGui.QPixmap.fromImage(p)


def main():
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
