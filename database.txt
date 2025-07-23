# hcc_backend_api/database.py
from sqlalchemy import create_engine, Column, Integer, String, Text, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, Session, relationship
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

# --- VERİTABANI AYARLARI ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./hcc_system.db"
engine = create_engine(
    SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False}
)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# --- VERİTABANI MODELLERİ (TABLOLAR) ---
# Bu sınıflar, SQL'deki tablolarınızı temsil eder.

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String)
    surname = Column(String)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    patients = relationship("Patient", back_populates="owner")

class Patient(Base):
    __tablename__ = "patients"
    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    surname = Column(String, index=True)
    age = Column(Integer)
    gender = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    owner = relationship("User", back_populates="patients")
    evaluations = relationship("Evaluation", back_populates="patient")

class Evaluation(Base):
    __tablename__ = "evaluations"
    id = Column(Integer, primary_key=True, index=True)
    evaluation_date = Column(DateTime, default=datetime.utcnow)
    patient_id = Column(Integer, ForeignKey("patients.id"))
    patient_details_json = Column(Text)
    api_result_json = Column(Text)
    patient = relationship("Patient", back_populates="evaluations")

# --- YARDIMCI FONKSİYON ---
def get_db():
    """Veritabanı oturumu (session) oluşturur ve yönetir."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
