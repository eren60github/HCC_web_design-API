# hcc_backend_api/main.py (İki Dosyalı Yapıya Uygun - Tamamlanmış Hali)
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Union
import json
from datetime import datetime
import pandas as pd
import io
from PIL import Image

# Yeni oluşturduğumuz database.py dosyasından gerekli her şeyi import ediyoruz
from database import Base, engine, get_db, User, Patient, Evaluation

# --- ŞİFRELEME ve DİĞER KÜTÜPHANELER ---
from passlib.context import CryptContext
import joblib
import tensorflow as tf
import numpy as np
import os
import uvicorn

# Veritabanı ve tabloları oluştur
Base.metadata.create_all(bind=engine)

# --- PYDANTIC MODELLERİ (API GİRDİ/ÇIKTI) ---
class UserCreate(BaseModel):
    name: Optional[str] = None
    surname: Optional[str] = None
    email: str
    password: str

class UserLogin(BaseModel):
    email: str
    password: str

class LabData(BaseModel):
    Yaş: float
    Cinsiyet: int
    Albumin: float
    ALP: float
    ALT: float
    AST: float
    BIL: float
    GGT: float

# --- AYARLAR ve UYGULAMA ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
LAB_MODEL_PATH = 'hcc_multi_model_xgboost.joblib'
LAB_SCALER_PATH = 'hcc_scaler_multi.joblib'
USG_MODEL_PATH = 'fibroz_vgg16_model.h5'
model_lab, scaler_lab, model_usg = None, None, None

app = FastAPI(title="HCC Erken Teşhis Sistemi API")

# CORS Ayarları
from fastapi.middleware.cors import CORSMiddleware
origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API BAŞLANGIÇ OLAYLARI ---
@app.on_event("startup")
async def load_models():
    global model_lab, scaler_lab, model_usg
    try:
        if os.path.exists(LAB_MODEL_PATH) and os.path.exists(LAB_SCALER_PATH):
            model_lab = joblib.load(LAB_MODEL_PATH)
            scaler_lab = joblib.load(LAB_SCALER_PATH)
            print("Lab modeli ve scaler başarıyla yüklendi.")
        if os.path.exists(USG_MODEL_PATH):
            model_usg = tf.keras.models.load_model(USG_MODEL_PATH, compile=False)
            print("USG modeli başarıyla yüklendi.")
    except Exception as e:
        print(f"HATA: Modeller yüklenirken bir sorun oluştu: {e}")

# --- KULLANICI ENDPOINT'LERİ ---
@app.post("/register", status_code=201)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user:
        raise HTTPException(status_code=400, detail="Bu e-posta adresi zaten kayıtlı.")
    hashed_password = pwd_context.hash(user.password)
    new_user = User(email=user.email, hashed_password=hashed_password, name=user.name, surname=user.surname)
    db.add(new_user); db.commit(); db.refresh(new_user)
    return {"message": f"Kullanıcı '{user.email}' başarıyla oluşturuldu."}

@app.post("/login")
def login_user(user: UserLogin, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if not db_user or not pwd_context.verify(user.password, db_user.hashed_password):
        raise HTTPException(status_code=401, detail="E-posta veya şifre hatalı.")
    return {"message": "Giriş başarılı!", "user_id": db_user.id, "user_name": db_user.name}

# --- YARDIMCI TAHMİN FONKSİYONLARI ---
async def predict_lab_risk(data: LabData):
    if model_lab is None or scaler_lab is None: raise HTTPException(status_code=500, detail="Lab modeli veya scaler yüklenmedi.")
    features_order_lab = ['Yaş', 'Cinsiyet', 'Albumin', 'ALP', 'ALT', 'AST', 'BIL', 'GGT']
    input_df_lab = pd.DataFrame([data.model_dump()], columns=features_order_lab)
    input_df_lab_scaled = scaler_lab.transform(input_df_lab)
    predictions_proba = model_lab.predict_proba(input_df_lab_scaled)[0]
    predicted_class_id = np.argmax(predictions_proba)
    disease_map = {0: "Sağlıklı", 1: "Hepatit", 2: "Fibröz", 3: "Siroz", 4: "HCC"}
    predicted_disease = disease_map.get(predicted_class_id, "Bilinmiyor")
    hcc_prob = predictions_proba[4] if len(predictions_proba) > 4 else 0.0
    risk_level = "Yüksek Risk" if hcc_prob >= 0.66 else "Orta Risk" if hcc_prob >= 0.33 else "Düşük Risk"
    return {"predicted_disease": predicted_disease, "hcc_probability": hcc_prob, "risk_level": risk_level}

async def predict_usg_fibrosis(file: UploadFile):
    if model_usg is None: raise HTTPException(status_code=500, detail="USG modeli yüklenmedi.")
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert('L').resize((224, 224))
    image_array = np.array(image)
    image_array = np.expand_dims(image_array, axis=-1)
    image_array_rgb = np.repeat(image_array, 3, axis=-1) / 255.0
    input_tensor = np.expand_dims(image_array_rgb, axis=0)
    predictions = model_usg.predict(input_tensor)
    predicted_class_id = np.argmax(predictions, axis=1)[0]
    fibrosis_labels = ['F0- Fibroz yok', 'F1- Hafif Fibroz', 'F2- Orta Fibroz', 'F3- Ağır Fibroz', 'F4- Siroz']
    return {"stage_label": fibrosis_labels[predicted_class_id], "stage_id": int(predicted_class_id)}

# --- MERKEZİ TAHMİN ENDPOINT'İ ---
@app.post("/evaluate_hcc_risk")
async def evaluate_hcc_risk(
    user_id: int = Form(...),
    patient_name: str = Form(...),
    patient_surname: str = Form(...),
    lab_data: str = Form(...),
    usg_file: Union[UploadFile, str] = Form(""),
    mri_file: Union[UploadFile, str] = Form(""),
    afp_value: Optional[float] = Form(None),
    alcohol_consumption: Optional[str] = Form(None),
    smoking_status: Optional[str] = Form(None),
    hcv_status: Optional[str] = Form(None),
    hbv_status: Optional[str] = Form(None),
    cancer_history_status: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    overall_risk_level = "Düşük Risk"
    detailed_report_summary = []
    mri_recommendation = False
    
    # 1. Lab Modelini Çalıştır
    lab_data_dict = json.loads(lab_data)
    lab_data_pydantic = LabData(**lab_data_dict)
    lab_result = await predict_lab_risk(lab_data_pydantic)
    detailed_report_summary.append(f"Laboratuvar Analizi: Tahmin Edilen Hastalık: {lab_result['predicted_disease']}, HCC Riski Seviyesi: {lab_result['risk_level']}")
    overall_risk_level = lab_result['risk_level']
    if overall_risk_level == "Yüksek Risk": mri_recommendation = True

    # 2. USG Görüntüsü Varsa USG Modelini Çalıştır
    if isinstance(usg_file, UploadFile):
        usg_result = await predict_usg_fibrosis(usg_file)
        detailed_report_summary.append(f"USG Görüntü Analizi: Karaciğer Fibrozis Evresi: {usg_result['stage_label']}")
        if usg_result["stage_id"] >= 3: # F3 veya F4 ise
            if overall_risk_level != "Yüksek Risk": overall_risk_level = "Yüksek Risk"
            mri_recommendation = True
        elif usg_result["stage_id"] >= 1 and overall_risk_level == "Düşük Risk": # F1 veya F2 ise
            overall_risk_level = "Orta Risk"

    # ... (Buraya AFP değeri ve diğer risk faktörlerine göre risk ayarlama mantığı eklenebilir) ...

    # 4. Nihai Öneri
    if mri_recommendation: final_recommendation = "HCC riski yüksek. Kesin tanı için MRI görüntülemesi ŞİDDETLE ÖNERİLİR."
    elif overall_risk_level == "Yüksek Risk": final_recommendation = "Yüksek düzeyde HCC riski. Uzman değerlendirmesi ve yakın takip önerilir."
    elif overall_risk_level == "Orta Risk": final_recommendation = "Orta düzeyde HCC riski. 6 ayda bir AFP ve USG ile yakın takip önerilir."
    else: final_recommendation = "HCC riski düşük. Rutin yıllık kontroller önerilir."

    api_result = {
        "overall_risk_level": overall_risk_level,
        "mri_recommendation": mri_recommendation,
        "final_recommendation": final_recommendation,
        "detailed_report_summary": detailed_report_summary
    }

    # Veritabanına kaydetme
    try:
        new_patient = Patient(name=patient_name, surname=patient_surname, age=lab_data_dict.get('Yaş'), gender="Erkek" if lab_data_dict.get('Cinsiyet') == 1 else "Kadın", user_id=user_id)
        db.add(new_patient); db.commit(); db.refresh(new_patient)
        
        patient_details_for_db = {"lab_data": lab_data_dict, "afp_value": afp_value, "risk_factors": {"alcohol": alcohol_consumption, "smoking": smoking_status}}
        new_evaluation = Evaluation(patient_id=new_patient.id, evaluation_date=datetime.utcnow(), patient_details_json=json.dumps(patient_details_for_db), api_result_json=json.dumps(api_result))
        db.add(new_evaluation); db.commit()
    except Exception as e:
        db.rollback()
        print(f"Veritabanı Kayıt Hatası: {e}")
        api_result["db_error"] = f"Sonuç üretildi ancak kaydedilemedi: {e}"

    return api_result

# --- API'yi Çalıştırma ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
