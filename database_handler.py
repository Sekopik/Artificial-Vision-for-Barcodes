from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Enum
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
import enum
from config import cfg

Base = declarative_base()

class StatusEnum(enum.Enum):
    SUCCESS = "SUCCESS"
    FAIL_DETECTION = "FAIL_DETECTION"
    FAIL_OCR = "FAIL_OCR"
    ERROR = "ERROR"

class Reading(Base):
    __tablename__ = 'lecturas'

    id = Column(Integer, primary_key=True)
    filename = Column(String, nullable=False)
    extracted_text = Column(String, nullable=True)
    confidence = Column(Float, default=0.0)
    status = Column(String, nullable=False) # Guardamos como string para simplicidad en SQLite
    timestamp = Column(DateTime, default=datetime.now)
    processing_time = Column(Float, default=0.0) # Segundos

class DatabaseHandler:
    def __init__(self):
        db_url = cfg.get('DATABASE', 'db_file')
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def save_event(self, filename, text, confidence, status, proc_time):
        session = self.Session()
        try:
            # Asegurar que status sea string
            status_str = status.value if isinstance(status, StatusEnum) else str(status)
            
            record = Reading(
                filename=filename,
                extracted_text=text,
                confidence=confidence,
                status=status_str,
                processing_time=proc_time
            )
            session.add(record)
            session.commit()
        except Exception as e:
            print(f"[DB ERROR] No se pudo guardar el registro: {e}")
            session.rollback()
        finally:
            session.close()