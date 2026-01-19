import cv2
import numpy as np
import re
import math
import os
from config import cfg

try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ImagePipeline:
    def __init__(self):
        self.use_yolo = cfg.get_bool('MODULES', 'enable_yolo_detection')
        self.yolo_conf = cfg.get_float('OCR_PARAMS', 'yolo_conf', 0.35)
        self.yolo_imgsz = cfg.get_int('OCR_PARAMS', 'yolo_imgsz', 1024)
        
        self.yolo_model = None
        if self.use_yolo:
            model_path = cfg.get('MODULES', 'yolo_model_path', 'best.pt')
            print(f"[INIT] Cargando modelo YOLO OBB: {model_path}...")
            if YOLO and os.path.exists(model_path):
                self.yolo_model = YOLO(model_path)
            else:
                print(f"[CRITICAL WARNING] No se encontró YOLO en {model_path}.")

        self.ocr_model = None
        if cfg.get('MODULES', 'ocr_engine') == 'RAPIDOCR':
            if RapidOCR:
                self.ocr_model = RapidOCR(text_score=0.3, use_angle_cls=False)

    def crop_box_from_background(self, image, filename="debug"):
        try:
            h_orig, w_orig = image.shape[:2]
            scale = 0.2
            small = cv2.resize(image, (0, 0), fx=scale, fy=scale)
            if len(small.shape) == 3: gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            else: gray = small
            
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            edges = cv2.Canny(blurred, 30, 100)
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=2)
            
            contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours: return image, image
            
            largest_c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_c) < (50 * 50): return image, image
            
            x_s, y_s, w_s, h_s = cv2.boundingRect(largest_c)
            x, y, w, h = int(x_s/scale), int(y_s/scale), int(w_s/scale), int(h_s/scale)
            
            padding = 50
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(w_orig - x, w + (padding * 2))
            h = min(h_orig - y, h + (padding * 2))
            
            cropped = image[y:y+h, x:x+w]
            debug_path = os.path.join(cfg.get('PATHS', 'debug_folder'), f"STEP0_ZOOM_{filename}")
            cv2.imwrite(debug_path, cropped)
            return cropped, image
        except Exception as e:
            return image, image

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)] 
        rect[2] = pts[np.argmax(s)] 
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)] 
        rect[3] = pts[np.argmax(diff)] 
        return rect

    def extract_obb_crop(self, image, obb_result):
        try:
            x_c, y_c, w, h, rotation_rad = obb_result.xywhr[0].cpu().numpy()
            rotation_deg = math.degrees(rotation_rad)
            
            w = w * 1.02
            h = h * 1.02
            
            rect = ((x_c, y_c), (w, h), rotation_deg)
            box = cv2.boxPoints(rect)
            box = np.array(box, dtype="int32")
            
            src_pts = self.order_points(box.astype("float32"))
            
            (tl, tr, br, bl) = src_pts
            widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
            widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
            maxWidth = max(int(widthA), int(widthB))

            heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
            heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
            maxHeight = max(int(heightA), int(heightB))

            dst_pts = np.array([
                [0, 0], [maxWidth - 1, 0],
                [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]], dtype="float32")
            
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
            return warped, box
        except Exception as e:
            print(f"[ERROR OBB] {e}")
            return None, None

    def find_best_sequence(self, raw_text):
        """
        LÓGICA V17: JERARQUÍA DE MÉTODOS
        Retorna: (código_candidato, es_valido_longitud, metodo_usado)
        metodo_usado: 2 (Estructurado - ALTA PRIORIDAD), 1 (Fuerza Bruta - BAJA PRIORIDAD), 0 (Nada)
        """
        text = raw_text.upper()
        replacements = {'O': '0', 'Z': '2', 'B': '8', 'S': '5'}
        for k, v in replacements.items():
            text = text.replace(k, v)
        
        final_candidate = ""
        method_score = 0 # 0=Fail, 1=Brute, 2=Structured

        # --- ESTRATEGIA A: ESTRUCTURADA (Orden 10 -> 12) ---
        chunks = re.findall(r'\d+', text)
        part_10 = None
        part_12 = None
        structured_failed = False

        for seq in chunks:
            l = len(seq)
            if 10 <= l <= 11 and part_10 is None:
                part_10 = seq
                continue
            if 12 <= l <= 14 and part_12 is None:
                if part_10 is not None:
                    part_12 = seq
                    break 
                else:
                    # Encontró el 12 antes que el 10 -> Orden Incorrecto
                    structured_failed = True
                    break
        
        if part_10 and part_12 and not structured_failed:
            combined = part_10 + part_12
            # Limpieza interna
            if combined.startswith('55') and len(combined) > 22: combined = combined[2:]
            if len(combined) == 23 and combined.endswith(('0', '5')): combined = combined[:-1]
            elif len(combined) == 24 and combined.endswith(('00', '10', '01')): combined = combined[:-2]
            
            if len(combined) == 22:
                # ¡JACKPOT! Estructura perfecta.
                return combined, True, 2 

        # --- ESTRATEGIA B: FUERZA BRUTA (Fallback) ---
        # Solo llegamos aquí si la estructurada falló o no dio 22 exactos.
        blob = text.replace(" ", "")
        blob_chunks = re.findall(r'\d+', blob)
        
        if blob_chunks:
            longest = max(blob_chunks, key=len)
            # Limpieza interna
            if longest.startswith('55') and len(longest) > 22: longest = longest[2:]
            if len(longest) == 23 and longest.endswith('0'): longest = longest[:-1]
            elif len(longest) == 24 and longest.endswith(('00', '10', '01')): longest = longest[:-2]
            
            if len(longest) == 22:
                return longest, True, 1 # Retornamos con prioridad baja (1)

        # Si todo falla, retornamos lo mejor que tengamos (Fuerza Bruta)
        if blob_chunks and not final_candidate:
            return max(blob_chunks, key=len), False, 0
            
        return "", False, 0

    def process_image(self, image, filename="debug"):
        zoomed_img, _ = self.crop_box_from_background(image, filename)
        
        if len(zoomed_img.shape) == 2:
            img_yolo = cv2.cvtColor(zoomed_img, cv2.COLOR_GRAY2BGR)
        else:
            img_yolo = zoomed_img.copy()

        if not self.yolo_model: return None, 0, "YOLO_FAIL", None
        results = self.yolo_model(img_yolo, imgsz=self.yolo_imgsz, conf=self.yolo_conf, verbose=False)
        
        if not results or not results[0].obb:
            print(f"[YOLO] Nada detectado en {filename}")
            return None, 0, "NO_DETECTION", None
        
        best_obb = None
        best_yolo_conf = 0.0
        for obb in results[0].obb:
            conf = float(obb.conf)
            if conf > best_yolo_conf:
                best_yolo_conf = conf
                best_obb = obb
        
        if best_obb is None: return None, 0, "LOW_CONFIDENCE", None

        crop, box_points = self.extract_obb_crop(img_yolo, best_obb)
        
        if box_points is not None:
            viz_img = img_yolo.copy()
            cv2.drawContours(viz_img, [box_points], 0, (0, 255, 0), 4)
            debug_viz_path = os.path.join(cfg.get('PATHS', 'debug_folder'), f"STEP0.5_YOLO_DETECTED_{filename}")
            cv2.imwrite(debug_viz_path, viz_img)

        if crop is None: return None, 0, "CROP_FAILED", None

        h_c, w_c = crop.shape[:2]
        if h_c > w_c:
            crop = cv2.rotate(crop, cv2.ROTATE_90_CLOCKWISE)

        if len(crop.shape) == 3: crop_gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else: crop_gray = crop

        debug_crop_path = os.path.join(cfg.get('PATHS', 'debug_folder'), f"STEP1_CROP_{filename}")
        cv2.imwrite(debug_crop_path, crop_gray)

        attempts = [
            {"angle": 0, "img": crop_gray},
            {"angle": 180, "img": cv2.rotate(crop_gray, cv2.ROTATE_180)}
        ]

        winner = {"text": "", "score": -1, "conf": 0, "angle": 0, "img": crop_gray}

        print(f"--- TORNEO OCR para {filename} ---")
        for attempt in attempts:
            img_input = attempt["img"]
            result, _ = self.ocr_model(img_input)
            
            if not result: continue
            
            full_text_raw = " ".join([line[1] for line in result])
            avg_conf = sum([line[2] for line in result]) / len(result)

            # Ahora recibimos también el 'method_score' (2=Estructurado, 1=Bruto)
            candidate_code, is_valid_len, method_score = self.find_best_sequence(full_text_raw)
            
            # --- SISTEMA DE PUNTUACIÓN V17 BLINDADO ---
            # Base: Confianza (0-100)
            score = avg_conf * 100 
            
            # 1. VALIDEZ DEL CÓDIGO (22 dígitos)
            if is_valid_len: 
                score += 2000 # Premiazo indiscutible
            
            # 2. MÉTODO ESTRUCTURADO (La joya de la corona)
            # Si se encontró respetando el orden 10->12, le damos prioridad sobre la fuerza bruta.
            if method_score == 2:
                score += 1000 
            
            # 3. ANCLA DE ORIENTACIÓN (Header) - FIX ESPACIOS
            # Quitamos espacios para que "CHILEX PRESS" sea "CHILEXPRESS" y matchee.
            header_keywords = ["CHILEXPRESS", "CHILEXPRE55"]
            raw_upper_nospace = full_text_raw.upper().replace(" ", "")
            
            if any(k in raw_upper_nospace for k in header_keywords):
                score += 3000 # GOD MODE: Si lees el header, estás al derecho. Fin.

            # 4. PENALIZACIONES
            if len(candidate_code) != 22: 
                score -= 500

            print(f"Angle {attempt['angle']}º | Method: {method_score} | Valid: {is_valid_len} | Score: {score:.1f}")

            if score > winner["score"]:
                winner.update({
                    "text": candidate_code,
                    "score": score,
                    "conf": avg_conf,
                    "angle": attempt["angle"],
                    "img": img_input
                })

        if winner["score"] < 100:
            print(f"[FAIL] Sin match válido. Intento: {winner['text']}")
            return None, winner["conf"], "OCR_FAIL", winner["img"]

        print(f"[WINNER] Ángulo: {winner['angle']}º | Código: {winner['text']}")
        return winner["text"], winner["conf"], "SUCCESS", winner["img"]
    