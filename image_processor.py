import cv2
import numpy as np
import re
import time
from config import cfg

# Imports condicionales para optimizar memoria si no se usan
try:
    from rapidocr_onnxruntime import RapidOCR
except ImportError:
    RapidOCR = None

try:
    from doctr.models import ocr_predictor
    from doctr.io import Document
except ImportError:
    ocr_predictor = None

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None

class ImagePipeline:
    def __init__(self):
        self.ocr_engine_type = cfg.get('MODULES', 'ocr_engine').upper()
        self.use_yolo = cfg.get_bool('MODULES', 'enable_yolo_detection')
        self.use_preprocess = cfg.get_bool('MODULES', 'enable_preprocessing')
        self.min_conf = cfg.get_float('OCR_PARAMS', 'min_confidence')
        self.min_len = cfg.get_int('OCR_PARAMS', 'min_length', 8)

        # Inicializar YOLO
        self.yolo_model = None
        if self.use_yolo:
            model_path = cfg.get('MODULES', 'yolo_model_path', 'yolov8n.pt')
            print(f"[INIT] Cargando modelo YOLO desde {model_path}...")
            if YOLO:
                self.yolo_model = YOLO(model_path)
            else:
                print("[ERROR] Ultralytics no está instalado.")

        # Inicializar OCR Engine
        self.ocr_model = None
        if self.ocr_engine_type == 'RAPIDOCR':
            print("[INIT] Cargando RapidOCR (ONNX)...")
            if RapidOCR:
                self.ocr_model = RapidOCR()
            else:
                raise ImportError("RapidOCR no instalado.")
        
        elif self.ocr_engine_type == 'DOCTR':
            print("[INIT] Cargando DocTR (Torch)...")
            if ocr_predictor:
                # pretrained=True descarga el modelo si no existe
                self.ocr_model = ocr_predictor(det_arch='db_resnet50', reco_arch='crnn_vgg16_bn', pretrained=True)
            else:
                raise ImportError("DocTR no instalado.")

    def preprocess_image(self, image):
        """
        Pipeline experto para:
        1. Eliminar brillos (plástico).
        2. Unir puntos (fuentes matriciales).
        3. Mejorar resolución.
        """
        if not self.use_preprocess:
            return image

        # 1. Escala de Grises
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
        # Fundamental para eliminar el "glare" del plástico retráctil.
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        contrast = clahe.apply(gray)

        # 3. Dilatación Morfológica Suave
        # Fundamental para etiquetas de farmacia (matriciales de puntos).
        # Engorda los puntos para que el OCR vea líneas continuas.
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(contrast, kernel, iterations=1)

        # 4. Upscale (2x)
        # Ayuda a los modelos OCR a detectar caracteres pequeños.
        h, w = dilated.shape
        upscaled = cv2.resize(dilated, (w * 2, h * 2), interpolation=cv2.INTER_CUBIC)

        return upscaled

    def detect_roi(self, image):
        """Recorta la etiqueta usando YOLO si está habilitado."""
        if not self.use_yolo or self.yolo_model is None:
            return image

        results = self.yolo_model(image, verbose=False)
        # Tomar la detección con mayor confianza
        best_box = None
        max_conf = 0.0

        for result in results:
            for box in result.boxes:
                if box.conf.item() > max_conf:
                    max_conf = box.conf.item()
                    best_box = box.xyxy.cpu().numpy()[0] # [x1, y1, x2, y2]

        if best_box is not None:
            x1, y1, x2, y2 = map(int, best_box)
            # Validar límites
            h, w = image.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            return image[y1:y2, x1:x2]
        
        return image

    def clean_text(self, text):
        """Post-procesado heurístico para errores comunes en industrial."""
        if not text: return ""
        
        # Mayúsculas
        text = text.upper()
        
        # Correcciones típicas OCR (Letra -> Número)
        replacements = {
            'O': '0', 'D': '0', 'Q': '0',
            'I': '1', 'L': '1', '|': '1',
            'Z': '2',
            'S': '5',
            'B': '8',
            'G': '6'
        }
        for char, digit in replacements.items():
            text = text.replace(char, digit)

        # Extraer solo dígitos seguidos
        # Buscamos la secuencia de dígitos más larga
        digits_groups = re.findall(r'\d+', text)
        if not digits_groups:
            return ""
        
        # Devolver el grupo más largo encontrado
        longest_seq = max(digits_groups, key=len)
        return longest_seq

    def extract_text(self, image):
        raw_text = ""
        avg_conf = 0.0

        if self.ocr_engine_type == 'RAPIDOCR':
            # RapidOCR devuelve: [[[[x1,y1]...], "texto", confianza], ...]
            # En rapidocr_onnxruntime >= 1.3.0, la llamada devuelve (result, elapse)
            result, _ = self.ocr_model(image)
            
            if result:
                # Filtrar y concatenar mejores candidatos
                candidates = []
                for line in result:
                    box, text, conf = line
                    if conf >= self.min_conf:
                        cleaned = self.clean_text(text)
                        if len(cleaned) >= self.min_len:
                            candidates.append((cleaned, conf))
                
                if candidates:
                    # Tomar el que tenga mayor longitud y buena confianza
                    best = max(candidates, key=lambda x: len(x[0]))
                    raw_text = best[0]
                    avg_conf = best[1]

        elif self.ocr_engine_type == 'DOCTR':
            # Doctr espera lista de arrays o tensores
            # DocTR devuelve un objeto Document
            try:
                res = self.ocr_model([image])
                json_output = res.export()
                
                words_found = []
                
                # Navegar la estructura JSON de Doctr
                for page in json_output['pages']:
                    for block in page['blocks']:
                        for line in block['lines']:
                            line_text = ""
                            line_conf = []
                            for word in line['words']:
                                if word['confidence'] >= self.min_conf:
                                    line_text += word['value']
                                    line_conf.append(word['confidence'])
                            
                            cleaned = self.clean_text(line_text)
                            if len(cleaned) >= self.min_len:
                                conf_val = sum(line_conf)/len(line_conf) if line_conf else 0
                                words_found.append((cleaned, conf_val))
                
                if words_found:
                    best = max(words_found, key=lambda x: len(x[0]))
                    raw_text = best[0]
                    avg_conf = best[1]

            except Exception as e:
                print(f"[ERROR DOCTR] {e}")

        return raw_text, avg_conf