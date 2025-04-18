import cv2
import numpy as np
from collections import deque
import imutils
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

class KalmanFilter:
    def __init__(self):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                               [0, 1, 0, 1],
                                               [0, 0, 1, 0],
                                               [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0],
                                              [0, 1, 0, 0],
                                              [0, 0, 1, 0],
                                              [0, 0, 0, 1]], np.float32) * 0.03
        self.initialized = False

    def update(self, point):
        measurement = np.array([[np.float32(point[0])], [np.float32(point[1])]])
        
        if not self.initialized:
            self.kalman.statePre = np.array([[np.float32(point[0])],
                                           [np.float32(point[1])],
                                           [0], [0]], np.float32)
            self.kalman.statePost = np.array([[np.float32(point[0])],
                                            [np.float32(point[1])],
                                            [0], [0]], np.float32)
            self.initialized = True
            
        prediction = self.kalman.predict()
        estimated = self.kalman.correct(measurement)
        
        return (int(estimated[0][0]), int(estimated[1][0]))

class MotionDetector:
    def __init__(self):
        self.frame_buffer = deque(maxlen=5)
        self.min_area = 50  # Hassas minimum alan
        self.max_area = 50000
        self.previous_frame = None
        self.frame_count = 0
        self.movement_area = None
        self.vertical_padding = 30
        self.movement_history = []
        self.history_length = 15
        self.no_movement_counter = 0
        
        # Hassasiyet parametreleri
        self.detection_threshold = 5
        self.blur_size = 1
        
        self.roi = None
        self.roi_padding = 10
        
        # Yoğun bölge ve uç nokta parametreleri (Tekrar Aktif)
        self.density_threshold = 0.5  # Orta seviye yoğunluk eşiği
        self.tip_region_height = 150
        self.tip_weight = 2.0  # Uç bölge ağırlığı
        self.min_continuous_frames = 2 # Esnek doğrulama
        self.movement_verification_buffer = deque(maxlen=5)
        
        self.classifier = RandomForestClassifier(n_estimators=100)
        self.training_data = []
        self.training_labels = []
        
        # Kalman Filtresi (Tekrar Aktif)
        self.kalman_filter = KalmanFilter()
        self.tracked_points = []
        self.main_movement_area = None

    def preprocess_video(self, video_path, sample_frames=100):
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

    def verify_movement(self, current_rect):
        """Hareketin gerçek olup olmadığını kontrol et (Tekrar Aktif)"""
        if not current_rect:
            self.movement_verification_buffer.append(None)
            return False
            
        self.movement_verification_buffer.append(current_rect)
        
        # Son birkaç frame'deki hareket konumlarını kontrol et
        valid_rects = [r for r in self.movement_verification_buffer if r is not None]
        if len(valid_rects) < self.min_continuous_frames:
            return False
            
        # Hareketin tutarlılığını kontrol et
        x_coords = [r[0] for r in valid_rects]
        y_coords = [r[1] for r in valid_rects]
        
        # Konum değişiminin tutarlı olup olmadığını kontrol et
        x_diff = max(x_coords) - min(x_coords)
        y_diff = max(y_coords) - min(y_coords)
        
        # Ani sıçramalar varsa hareketi reddet (Esnek tolerans)
        return x_diff < 100 and y_diff < 100

    def detect_vertical_movement(self, frame):
        # ROI Kullanımı
        frame_roi = frame
        roi_x, roi_y = 0, 0
        if self.roi:
            x, y, w, h = self.roi
            frame_roi = frame[y:y+h, x:x+w]
            roi_x, roi_y = x, y
        
        gray = cv2.cvtColor(frame_roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (self.blur_size, self.blur_size), 0)
        
        if self.previous_frame is None:
            self.previous_frame = gray
            return frame
            
        frame_diff = cv2.absdiff(self.previous_frame, gray)
        _, thresh = cv2.threshold(frame_diff, self.detection_threshold, 255, cv2.THRESH_BINARY)
        
        # Dikey hareketi vurgula
        kernel_vertical = np.ones((5, 1), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_vertical)
        thresh = cv2.dilate(thresh, kernel_vertical, iterations=1)
        
        contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = imutils.grab_contours(contours)
        
        # *** Filtreleme ve Yoğunluk Analizi (Tekrar Aktif) ***
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_area < area < self.max_area:
                _, y_coords = contour.reshape(-1, 2).T
                bottom_y = np.max(y_coords)
                valid_contours.append((area, bottom_y, contour))
        
        # Alt noktaya göre sırala
        valid_contours.sort(key=lambda x: x[1], reverse=True)
        
        height, width = frame.shape[:2]
        density_map = np.zeros((height, width), dtype=np.float32)
        
        movement_rects = []
        current_points = []
        
        bottom_region = height - self.tip_region_height
        
        # Konturları işle ve yoğunluk haritası oluştur
        for _, bottom_y, contour in valid_contours[:10]:
            x_c, y_c, w_c, h_c = cv2.boundingRect(contour)
            
            # ROI ofseti ekle
            x_global = x_c + roi_x
            y_global = y_c + roi_y
            
            weight = self.tip_weight if y_global + h_c > bottom_region else 1.0
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"]) + roi_x # Global cx
                cy = int(M["m01"] / M["m00"]) + roi_y # Global cy
                current_points.append((cx, cy))
                
                cv2.rectangle(density_map, (x_global, y_global), (x_global + w_c, y_global + h_c), weight, -1)
            
            movement_rects.append((x_global, y_global, w_c, h_c, weight))
        
        # En yoğun bölgeyi bul ve filtrele
        final_rect_to_draw = None
        if len(movement_rects) > 0:
            density_map = cv2.GaussianBlur(density_map, (15, 15), 0)
            density_map = cv2.normalize(density_map, None, 0, 1, cv2.NORM_MINMAX)
            
            # Üst bölgeyi bastır
            density_map[:-self.tip_region_height] *= 0.5
            
            max_val = np.max(density_map)
            if max_val > self.density_threshold:
                y_dense, x_dense = np.unravel_index(np.argmax(density_map), density_map.shape)
                
                # Yoğun bölgedeki en alttaki dikdörtgeni bul
                dense_rects = []
                for rect in movement_rects:
                    rx, ry, rw, rh, weight = rect
                    # Yoğunluk merkezine yakınlık kontrolü
                    if abs(x_dense - (rx + rw/2)) < rw*1.5 and abs(y_dense - (ry + rh/2)) < rh*1.5:
                        if ry + rh > bottom_region: # Alt bölgedeyse öncelik ver
                            dense_rects.insert(0, (rx, ry, rw, rh))
                
                if dense_rects:
                    # En alttaki doğrulanmış dikdörtgeni seç
                    for r in dense_rects:
                        if self.verify_movement(r):
                            final_rect_to_draw = r
                            break # İlk doğrulanmış olanı al
        
        # Sadece son, doğrulanmış hareketi çiz
        if final_rect_to_draw:
            x, y, w, h = final_rect_to_draw
            tip_point = (int(x + w/2), int(y + h))
            
            predicted_point = self.kalman_filter.update(tip_point)
            self.tracked_points.append(predicted_point)
            if len(self.tracked_points) > 5:
                self.tracked_points.pop(0)
                
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2) # Yeşil
            cv2.circle(frame, predicted_point, 2, (0, 0, 255), -1) # Kırmızı
        else:
            # Eğer doğrulanan hareket yoksa, doğrulama buffer'ını temizle
            self.verify_movement(None)

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