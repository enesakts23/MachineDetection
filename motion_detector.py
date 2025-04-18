import cv2
import numpy as np
from collections import deque
import imutils
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class MotionDetector:
    def __init__(self):
        self.frame_buffer = deque(maxlen=5)
        self.min_area = 500  # Daha küçük alanları da algıla
        self.max_area = 50000
        self.previous_frame = None
        self.frame_count = 0
        self.movement_area = None
        self.vertical_padding = 150
        self.movement_history = []
        self.history_length = 5
        self.no_movement_counter = 0
        
        # Hassasiyet parametreleri
        self.detection_threshold = 10  # Daha hassas threshold
        self.blur_size = 3  # Daha az blur
        
        self.roi = None  # Tespit edilen hareket bölgesi
        self.roi_padding = 50  # ROI için ek padding
        
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.training_data = []
        self.training_labels = []

    def preprocess_video(self, video_path, sample_frames=50):
        """Video başlamadan önce hareketli bölgeyi tespit et"""
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = np.linspace(0, total_frames-1, sample_frames, dtype=int)
        
        frames = []
        diffs = []
        
        # Frame'leri topla
        for idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
                frames.append(gray)
                
                if len(frames) > 1:
                    diff = cv2.absdiff(frames[-1], frames[-2])
                    diffs.append(diff)
        
        cap.release()
        
        if not diffs:
            return None
            
        # Tüm frame farklarını birleştir
        motion_map = np.zeros_like(diffs[0], dtype=np.float32)
        for diff in diffs:
            _, thresh = cv2.threshold(diff, self.detection_threshold, 255, cv2.THRESH_BINARY)
            motion_map += thresh.astype(np.float32)
        
        # Hareket haritasını normalize et
        motion_map = (motion_map / len(diffs)).astype(np.uint8)
        
        # Dikey hareketi vurgula
        kernel_vertical = np.ones((7, 3), np.uint8)
        motion_map = cv2.morphologyEx(motion_map, cv2.MORPH_OPEN, kernel_vertical)
        motion_map = cv2.dilate(motion_map, kernel_vertical, iterations=2)
        
        # En aktif bölgeyi bul
        contours = cv2.findContours(motion_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        if contours:
            # En büyük konturu bul
            max_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(max_contour)
            
            # ROI'yi genişlet
            x = max(0, x - self.roi_padding)
            y = max(0, y - self.roi_padding)
            w = min(motion_map.shape[1] - x, w + 2 * self.roi_padding)
            h = min(motion_map.shape[0] - y, h + 2 * self.roi_padding)
            
            self.roi = (x, y, w, h)
            return True
        
        return False

    def detect_vertical_movement(self, frame):
        # ROI'yi kullan
        if self.roi:
            x, y, w, h = self.roi
            frame_roi = frame[y:y+h, x:x+w]
        else:
            frame_roi = frame
        
        # Frame'i griye çevir
        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return frame
            
        # Frame farkını hesapla
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        
        # Threshold uygula
        _, thresh = cv2.threshold(frame_diff, self.detection_threshold, 255, cv2.THRESH_BINARY)
        
        # Dikey hareketi vurgula
        kernel_vertical = np.ones((7, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_vertical)
        thresh = cv2.dilate(thresh, kernel_vertical, iterations=2)
        
        # Konturları bul
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        movement_detected = False
        
        if len(contours) > 0:
            # Tüm konturları birleştir
            all_points = np.vstack([cont.reshape(-1, 2) for cont in contours])
            x_local, y_local, w_local, h_local = cv2.boundingRect(all_points)
            
            # ROI koordinatlarını global koordinatlara çevir
            if self.roi:
                roi_x, roi_y, _, _ = self.roi
                x = x_local + roi_x
                y = y_local + roi_y
                w = w_local
                h = h_local
            else:
                x, y, w, h = x_local, y_local, w_local, h_local
            
            area = w * h
            if self.min_area < area < self.max_area:
                movement_detected = True
                
                # Dikey padding ekle
                y = max(0, y - self.vertical_padding)
                h = min(frame.shape[0] - y, h + 2 * self.vertical_padding)
                
                # Hareket geçmişine ekle
                self.movement_history.append((x, y, w, h))
                if len(self.movement_history) > self.history_length:
                    self.movement_history.pop(0)
                
                # Hareket alanını güncelle
                if self.movement_area is None:
                    self.movement_area = (x, y, w, h)
                else:
                    mx, my, mw, mh = self.movement_area
                    # Yeni konuma daha fazla ağırlık ver
                    self.movement_area = (
                        int(0.1 * mx + 0.9 * x),
                        int(0.1 * my + 0.9 * y),
                        int(0.1 * mw + 0.9 * w),
                        int(0.1 * mh + 0.9 * h)
                    )
                self.no_movement_counter = 0
        
        if not movement_detected:
            self.no_movement_counter += 1
            if self.no_movement_counter > 2 and self.movement_area is not None:
                mx, my, mw, mh = self.movement_area
                # Hızlı küçültme
                self.movement_area = (
                    mx,
                    my,
                    int(mw * 0.6),  # Daha da hızlı küçült
                    int(mh * 0.6)
                )
                
                if mw * 0.6 < 50 or mh * 0.6 < 50:
                    self.movement_area = None
                    self.movement_history.clear()
        
        # Görselleştirme
        if self.movement_area is not None:
            mx, my, mw, mh = self.movement_area
            
            # Hareket yoğunluğu
            movement_intensity = np.sum(thresh) / 255.0
            
            # Ana dikdörtgen
            cv2.rectangle(frame, (int(mx), int(my)), (int(mx + mw), int(my + mh)), (0, 255, 0), 2)
            
            # Debug bilgisi
            cv2.putText(frame, f"Yogunluk: {int(movement_intensity)}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ROI'yi göster
        if self.roi:
            rx, ry, rw, rh = self.roi
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)
        
        # Debug görüntüsü
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
    video_path = "pressmachine4.mp4"
    detector = MotionDetector()
    
    print("Hareketli bölge tespit ediliyor...")
    detector.preprocess_video(video_path)
    print("Tespit tamamlandı, video başlatılıyor...")
    
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