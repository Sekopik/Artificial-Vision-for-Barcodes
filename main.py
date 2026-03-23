import os
import time
import threading
import queue
import shutil
import cv2
import logging
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from config import cfg
from image_processor import ImagePipeline
from database_handler import DatabaseHandler, StatusEnum

logging.basicConfig(
    filename=cfg.get('PATHS', 'log_file'),
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class FileProcessor(threading.Thread):
    def __init__(self, job_queue, pipeline, db):
        super().__init__()
        self.job_queue = job_queue
        self.pipeline = pipeline
        self.db = db
        self.daemon = True

    def run(self):
        print("[WORKER] Esperando imágenes...")
        while True:
            path = self.job_queue.get()
            if path is None: break

            filename = os.path.basename(path)
            print(f"\n>>> PROCESANDO: {filename}")
            start_time = time.time()
            
            try:
                # Lectura robusta de imagen
                img = cv2.imread(path)
                if img is None:
                    print("[ERROR] No se pudo leer la imagen.")
                    self.move_file(path, cfg.get('PATHS', 'error_folder'))
                    self.job_queue.task_done()
                    continue

                # --- LLAMADA AL NUEVO PIPELINE ---
                extracted_text, conf, status_msg, debug_img = self.pipeline.process_image(img, filename)
                
                duration = time.time() - start_time

                # Definir estado para DB
                if status_msg == "SUCCESS":
                    final_status = StatusEnum.SUCCESS
                    dest_folder = cfg.get('PATHS', 'processed_folder')
                    print(f"✅ ÉXITO: {extracted_text}")
                else:
                    final_status = StatusEnum.FAIL_OCR if status_msg == "OCR_FAIL" else StatusEnum.FAIL_DETECTION
                    dest_folder = cfg.get('PATHS', 'error_folder')
                    print(f"❌ FALLO: {status_msg}")

                # Guardar imagen debug del ganador (o el mejor intento fallido)
                if debug_img is not None:
                    debug_path = os.path.join(cfg.get('PATHS', 'debug_folder'), f"FINAL_{filename}")
                    cv2.imwrite(debug_path, debug_img)

                # Base de datos
                self.db.save_event(filename, extracted_text, float(conf), final_status, duration)
                
                # Mover archivo original
                self.move_file(path, dest_folder)

            except Exception as e:
                print(f"[CRASH] Error procesando {filename}: {e}")
                logging.error(f"Error {filename}", exc_info=True)
                self.move_file(path, cfg.get('PATHS', 'error_folder'))
            finally:
                self.job_queue.task_done()

    def move_file(self, src, folder):
        if not os.path.exists(src): return
        os.makedirs(folder, exist_ok=True)
        shutil.move(src, os.path.join(folder, os.path.basename(src)))

class NewImageHandler(FileSystemEventHandler):
    def __init__(self, queue):
        self.queue = queue
    def on_created(self, event):
        if not event.is_directory and event.src_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif')):
            time.sleep(0.5) # Esperar escritura SO
            self.queue.put(event.src_path)

def main():
    print("=== SISTEMA CHILEXPRESS OBB V3 (PRODUCCIÓN) ===")
    
    # Init Pipeline y DB
    db = DatabaseHandler()
    pipeline = ImagePipeline() # Carga YOLO y RapidOCR
    
    # Cola y Workers
    job_queue = queue.Queue()
    worker = FileProcessor(job_queue, pipeline, db)
    worker.start()

    # Watchdog
    observer = Observer()
    observer.schedule(NewImageHandler(job_queue), cfg.get('PATHS', 'input_folder'), recursive=False)
    observer.start()

    print(f"[RUNNING] Escuchando: {cfg.get('PATHS', 'input_folder')}")
    try:
        while True: time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        job_queue.put(None)
    observer.join()

if __name__ == "__main__":
    main()