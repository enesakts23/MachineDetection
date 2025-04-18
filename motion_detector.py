import cv2
import numpy as np
from collections import deque
import imutils
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class MotionDetector:
    def __init__(self):
        self.frame_buffer = deque(maxlen=5)  # Daha kısa buffer
        self.min_area = 100  # Çok daha düşük minimum alan
        self.previous_frame = None
        self.frame_count = 0
        self.movement_area = None
        self.vertical_padding = 150  # Daha fazla dikey padding
        self.movement_history = []  # Hareket geçmişi
        self.history_length = 10  # Son 10 hareketin geçmişi
        
        self.min_area = 2000      
        self.max_area = 50000     
        
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.training_data = []
        self.training_labels = []
        self.roi = None
        self.selecting_roi = False

    def detect_vertical_movement(self, frame):
        # Frame'i griye çevir ve ön işleme
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)  # Daha az blur
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return frame
            
        # Frame farkını bul ve threshold uygula
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 15, 255, cv2.THRESH_BINARY)  # Daha düşük threshold
        
        # Morfolojik işlemler
        kernel_dilate = np.ones((5, 7), np.uint8)  # Dikey yönde daha büyük kernel
        kernel_erode = np.ones((3, 3), np.uint8)
        
        thresh = cv2.erode(thresh, kernel_erode, iterations=1)
        thresh = cv2.dilate(thresh, kernel_dilate, iterations=2)
        
        # Konturları bul
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # Tüm konturları birleştir
        if len(contours) > 0:
            all_points = np.vstack([cont.reshape(-1, 2) for cont in contours])
            x, y, w, h = cv2.boundingRect(all_points)
            
            # Çok küçük alanları filtrele
            area = w * h
            if area > self.min_area:
                # Dikey yönde genişlet
                y = max(0, y - self.vertical_padding)
                h = min(frame.shape[0] - y, h + 2 * self.vertical_padding)
                
                # Hareket geçmişine ekle
                self.movement_history.append((x, y, w, h))
                if len(self.movement_history) > self.history_length:
                    self.movement_history.pop(0)
                
                # Son hareketlerin ortalamasını al
                if len(self.movement_history) > 0:
                    avg_x = int(np.mean([m[0] for m in self.movement_history]))
                    avg_y = int(np.mean([m[1] for m in self.movement_history]))
                    avg_w = int(np.mean([m[2] for m in self.movement_history]))
                    avg_h = int(np.mean([m[3] for m in self.movement_history]))
                    
                    # Hareket alanını yumuşak geçişle güncelle
                    if self.movement_area is None:
                        self.movement_area = (avg_x, avg_y, avg_w, avg_h)
                    else:
                        mx, my, mw, mh = self.movement_area
                        self.movement_area = (
                            int(0.8 * mx + 0.2 * avg_x),
                            int(0.8 * my + 0.2 * avg_y),
                            int(0.8 * mw + 0.2 * avg_w),
                            int(0.8 * mh + 0.2 * avg_h)
                        )
        
        # Hareket alanını çiz
        if self.movement_area is not None:
            mx, my, mw, mh = self.movement_area
            
            # Hareket yoğunluğunu göster
            movement_intensity = np.sum(thresh) / 255.0
            color = (0, min(255, movement_intensity/100), 0)
            
            # Ana dikdörtgen
            cv2.rectangle(frame, (mx, my), (mx + mw, my + mh), (0, 165, 255), 2)
            
            # Hareket yönünü gösteren oklar
            mid_x = mx + mw // 2
            arrow_length = 30
            cv2.arrowedLine(frame, (mid_x, my + mh), (mid_x, my + mh - arrow_length), 
                           color, 2, tipLength=0.3)
            cv2.arrowedLine(frame, (mid_x, my), (mid_x, my + arrow_length), 
                           color, 2, tipLength=0.3)
            
            # Debug bilgisi
            cv2.putText(frame, f"Yogunluk: {int(movement_intensity)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Görsel debug için threshold'u küçük pencerede göster
        debug_thresh = cv2.resize(thresh, (160, 120))
        frame[10:130, frame.shape[1]-170:frame.shape[1]-10] = \
            cv2.cvtColor(debug_thresh, cv2.COLOR_GRAY2BGR)
        
        self.previous_frame = gray
        return frame

    def is_repetitive_motion(self, current_centroid):
        if len(self.motion_history) < self.motion_history.maxlen:
            return False

        # Son birkaç frame'deki hareket noktalarını analiz et
        repetition_count = 0
        total_checks = 0
        
        for past_centroids in self.motion_history:
            for past_centroid in past_centroids:
                # Merkez noktalar arasındaki mesafeyi kontrol et
                distance = np.sqrt((current_centroid[0] - past_centroid[0])**2 + 
                                 (current_centroid[1] - past_centroid[1])**2)
                
                if distance < 30:  # Benzer konumda hareket
                    repetition_count += 1
                total_checks += 1

        if total_checks == 0:
            return False

        repetition_ratio = repetition_count / total_checks
        return repetition_ratio > self.pattern_threshold

    def set_roi(self, roi):
        self.roi = roi

    def extract_features(self, frame, regions):
        features = []
        for (x, y, w, h) in regions:
            roi = frame[y:y + h, x:x + w]
            if roi.size == 0:
                continue
            # Basit özellik çıkarımı
            mean_color = np.mean(roi, axis=(0, 1))
            std_color = np.std(roi, axis=(0, 1))
            features.append(np.concatenate([mean_color, std_color]))
        return features

    def add_training_sample(self, frame, regions, label):
        features = self.extract_features(frame, regions)
        for feature in features:
            self.training_data.append(feature)
            self.training_labels.append(label)

    def train_model(self):
        if len(self.training_data) == 0:
            print("Eğitim verisi bulunamadı!")
            return False

        X = np.array(self.training_data)
        y = np.array(self.training_labels)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        self.classifier.fit(X_train, y_train)
        score = self.classifier.score(X_test, y_test)
        print(f"Model doğruluk oranı: {score:.2f}")
        return True

    def save_model(self, filename="motion_model.pkl"):
        with open(filename, 'wb') as f:
            pickle.dump(self.classifier, f)

    def load_model(self, filename="motion_model.pkl"):
        with open(filename, 'rb') as f:
            self.classifier = pickle.load(f)

def main():
    video_path = "pressmachine2.mp4"
    detector = MotionDetector()
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Video açılamadı!")
        return

    cv2.namedWindow("Makine Hareketi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Makine Hareketi", 800, 600)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = imutils.resize(frame, width=800)
        processed_frame = detector.detect_vertical_movement(frame)
        cv2.imshow("Makine Hareketi", processed_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 