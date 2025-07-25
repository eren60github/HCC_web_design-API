# hcc_backend_api/main.py

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

# TensorFlow ve Gemini kütüphanelerini içe aktar
import google.generativeai as genai

# Uyarıları gizle (isteğe bağlı)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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
model_lab, scaler_lab, model_usg, vlm_model = None, None, None, None # VLM modelini de ekledik

# Sınıf isimleri (Keras modelinin tahmin çıktıları için)
CLASS_NAMES = ['F0- Fibroz yok', 'F1- Hafif Fibroz', 'F2- Orta Fibroz', 'F3- Ağır Fibroz', 'F4- Siroz']

app = FastAPI(title="HCC Erken Teşhis Sistemi API")

# CORS Ayarları
from fastapi.middleware.cors import CORSMiddleware
origins = ["http://localhost", "http://localhost:3000"] # Frontend'inizin çalıştığı portu eklediğinizden emin olun
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- API BAŞLANGIÇ OLAYLARI ---
@app.on_event("startup")
async def load_models():
    global model_lab, scaler_lab, model_usg, vlm_model # VLM modelini de global yaptık
    try:
        if os.path.exists(LAB_MODEL_PATH) and os.path.exists(LAB_SCALER_PATH):
            model_lab = joblib.load(LAB_MODEL_PATH)
            scaler_lab = joblib.load(LAB_SCALER_PATH)
            print("✅ Lab modeli ve scaler başarıyla yüklendi.")
        else:
            print(f"UYARI: Lab modeli ({LAB_MODEL_PATH}) veya scaler ({LAB_SCALER_PATH}) bulunamadı.")

        if os.path.exists(USG_MODEL_PATH):
            model_usg = tf.keras.models.load_model(USG_MODEL_PATH, compile=False)
            print(f"✅ USG modeli başarıyla yüklendi: {USG_MODEL_PATH}")
        else:
            print(f"UYARI: USG modeli ({USG_MODEL_PATH}) bulunamadı.")
            
        # Gemini API anahtarını yükle ve VLM modelini yapılandır
        api_dosya_yolu = 'api_key.txt' 
        try:
            with open(api_dosya_yolu, 'r') as f:
                GEMINI_API_KEY = f.read().strip()
            genai.configure(api_key=GEMINI_API_KEY)
            vlm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            print("✅ Gemini 'gemini-1.5-flash-latest' modeli başarıyla yapılandırıldı.")
        except FileNotFoundError:
            print(f"HATA: '{api_dosya_yolu}' dosyası bulunamadı! VLM modeli kullanılamayacak.")
            vlm_model = None # Model bulunamazsa None olarak ayarla
        except Exception as e:
            print(f"HATA: Gemini modeli yapılandırılamadı. Hata: {e}")
            vlm_model = None # Hata durumunda None olarak ayarla

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
    # bcrypt Attribute Error'ı bazen buradan kaynaklanabiliyor.
    # Geçici çözüm olarak bazen bcrypt sürümünü düşürmek gerekebilir (pip install "bcrypt==4.0.1")
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

async def predict_usg_fibrosis(file_bytes: bytes):
    """Verilen görüntü baytları için fibroz evresini tahmin eder."""
    if model_usg is None: raise HTTPException(status_code=500, detail="USG modeli yüklenmedi.")
    try:
        # Görüntüyü yükle, gri tona çevir (L), 224x224 boyutuna yeniden boyutlandır
        # modeliniz siyah-beyaz görüntüleri bekliyorsa 'L', renkli bekliyorsa 'RGB' kullanın
        # VGG16 genellikle RGB bekler, bu yüzden tekrar 3 kanala genişletiyoruz.
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize((224, 224))
        image_array = np.array(image)
        image_array = image_array / 255.0 # Normalizasyon
        input_tensor = np.expand_dims(image_array, axis=0) # Batch boyutu ekle
        
        predictions = model_usg.predict(input_tensor, verbose=0)
        predicted_class_id = np.argmax(predictions, axis=1)[0]
        return {"stage_label": CLASS_NAMES[predicted_class_id], "stage_id": int(predicted_class_id)}
    except Exception as e:
        # Hata durumunda daha açıklayıcı bir mesaj döndür
        raise HTTPException(status_code=500, detail=f"USG tahmin hatası: {e}. Lütfen geçerli bir görüntü dosyası yüklediğinizden emin olun.")

async def generate_radiology_report_vlm(image_bytes: bytes, predicted_stage_label: str):
    """Görüntü baytları ve tahmin edilen evreye göre VLM'den rapor oluşturur."""
    if vlm_model is None: return "VLM raporu oluşturulamadı: Gemini modeli yüklenmedi veya bir hata oluştu."
    
    prompt_template = f"""
    # ROL VE BAĞLAM
    Sen, uzman bir radyolog tarafından geliştirilen bir yapay zeka modelinin sonuçlarını doğrulayan ve açıklayan bir yapay zeka asistanısın. Bu model, aşağıda sunulan görüntü için **{predicted_stage_label}** sonucunu verdi.

    # GÖREV
    Senin görevin, bir uzman gibi davranarak bu tahmini doğrulamaktır. Görüntüyü dikkatlice incele ve **sadece {predicted_stage_label}** evresiyle uyumlu olan görsel kanıtları listeleyen structure bir rapor yaz. Amacın, modelin verdiği kararın GÖRSEL GEREKÇESİNİ oluşturmaktır.

    # İSTENEN RAPOR FORMATI
    Lütfen raporunu aşağıdaki maddelere sadık kalarak oluştur:
    **Hasta Görüntüsü Analizi**\\n---------------------------------\\n**Ön-Tahmin (Referans Model):** {predicted_stage_label}\\n**Görsel Bulgular (Tahmini Destekleyen):**\\n* **Karaciğer Yüzeyi:**\\n* **Karaciğer Parankimi (Dokusu):**\\n* **Vasküler Yapılar:**\\n* **Diğer İlgili Bulgular:**\\n**Sonuç:**
    """
    try:
        img_for_vlm = Image.open(io.BytesIO(image_bytes))
        # VLM modeline renkli görüntü göndermek gerekebilir (VGG16 gibi modeller RGB input bekler)
        if img_for_vlm.mode != 'RGB':
            img_for_vlm = img_for_vlm.convert('RGB')

        # generate_content_async asenkron çağrı için gereklidir
        response = await vlm_model.generate_content_async([prompt_template, img_for_vlm])
        
        # Gemini'den gelen Markdown formatını HTML'e daha uygun hale getirelim
        return response.text.replace('\\n', '<br>').replace('**', '<b>').replace('**', '</b>')
    except Exception as e:
        print(f"VLM Rapor oluşturma hatası: {e}") # Backend konsoluna hatayı yazdır
        return f"VLM Rapor oluşturma hatası: {e}"


# --- MERKEZİ TAHMİN ENDPOINT'İ ---
@app.post("/evaluate_hcc_risk")
async def evaluate_hcc_risk(
    user_id: int = Form(...),
    patient_name: str = Form(...),
    patient_surname: str = Form(...),
    lab_data: str = Form(...),
    usg_file: Optional[UploadFile] = File(None), # Default olarak None, isteğe bağlı olduğunu belirtir
    mri_file: Optional[UploadFile] = File(None), # Default olarak None
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
    vlm_radiology_report = None # Bu değişkenin başlangıç değeri None

    print(f"\n--- evaluate_hcc_risk başlıyor ---")
    print(f"Alınan user_id: {user_id}, patient_name: {patient_name}, patient_surname: {patient_surname}")
    
    # Lab verisini parse etmeden önce kontrol et
    try:
        lab_data_dict = json.loads(lab_data)
        print(f"Parsed lab_data_dict: {lab_data_dict}")
    except json.JSONDecodeError as e:
        print(f"HATA: lab_data JSON parse edilirken hata oluştu: {e}")
        raise HTTPException(status_code=400, detail=f"Geçersiz Lab Verisi: {e}")

    # 1. Lab Modelini Çalıştır
    lab_data_pydantic = LabData(**lab_data_dict)
    lab_result = await predict_lab_risk(lab_data_pydantic)
    detailed_report_summary.append(f"Laboratuvar Analizi: Tahmin Edilen Hastalık: {lab_result['predicted_disease']}, HCC Riski Seviyesi: {lab_result['risk_level']}")
    overall_risk_level = lab_result['risk_level']
    if overall_risk_level == "Yüksek Risk": mri_recommendation = True

    print(f"Lab sonucu: {lab_result}")

    # 2. USG Görüntüsü Varsa USG Modelini Çalıştır ve VLM Raporu Oluştur
    if usg_file and usg_file.filename != '': # usg_file'ın gerçekten yüklü olup olmadığını kontrol et
        print(f"USG Dosyası alındı: {usg_file.filename}")
        try:
            contents = await usg_file.read() # Dosya içeriğini oku
            # Dosyayı bir kere okuduktan sonra resetlemek gerekebilir
            # Eğer aynı dosyayı birden fazla fonksiyonda kullanacaksanız
            # usg_file.seek(0)
            print(f"USG Dosyası boyutu: {len(contents)} bayt") # Check if content is read
            
            # Keras modeli ile fibroz evresi tahmini yap
            usg_result = await predict_usg_fibrosis(contents)
            detailed_report_summary.append(f"USG Görüntü Analizi: Karaciğer Fibrozis Evresi: {usg_result['stage_label']}")
            print(f"USG Keras modeli tahmini: {usg_result['stage_label']}")
            
            # VLM Modelini Çalıştır (sadece ultrason için)
            vlm_radiology_report = await generate_radiology_report_vlm(contents, usg_result['stage_label'])
            print(f"VLM Raporu Oluşturuldu (ilk 100 karakter): {vlm_radiology_report[:100]}...") # Raporun tamamı çok uzun olabilir

            if usg_result["stage_id"] >= 3: # F3 veya F4 ise
                if overall_risk_level != "Yüksek Risk": overall_risk_level = "Yüksek Risk"
                mri_recommendation = True
            elif usg_result["stage_id"] >= 1 and overall_risk_level == "Düşük Risk": # F1 veya F2 ise
                overall_risk_level = "Orta Risk"
        except Exception as e:
            print(f"HATA: USG veya VLM işleme sırasında hata oluştu: {e}")
            vlm_radiology_report = f"VLM Raporu Oluşturulamadı (Hata: {e}). Lütfen ultrason görüntüsünü kontrol edin." # Frontend'e hata mesajı gönder

    else:
        print("USG Dosyası yüklenmedi (usg_file is None veya boş filename). VLM raporu oluşturulmayacak.")
        vlm_radiology_report = "Ultrason görüntüsü yüklenmediği için VLM raporu oluşturulmadı."


    # MRI dosyasını işleme (VLM raporu oluşturmuyor)
    if mri_file and mri_file.filename != '':
        print(f"MRI Dosyası alındı: {mri_file.filename}. MRI modeli entegrasyonu henüz yok.")
        detailed_report_summary.append(f"MRI Görüntüsü Yüklendi: {mri_file.filename} (Analiz henüz mevcut değil).")


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
        "detailed_report_summary": detailed_report_summary,
        "vlm_radiology_report": vlm_radiology_report # VLM raporunu API sonucuna ekledik
    }
    
    print(f"--- evaluate_hcc_risk sonu ---")
    print(f"Dönülen nihai api_result'ın anahtarları: {api_result.keys()}")
    print(f"Dönülen nihai api_result'ta 'vlm_radiology_report' var mı?: {'vlm_radiology_report' in api_result}")
    print(f"Dönülen nihai api_result'ta 'vlm_radiology_report' değeri NULL mı?: {api_result['vlm_radiology_report'] is None}")
    if 'vlm_radiology_report' in api_result and api_result['vlm_radiology_report']:
        print(f"Dönülen nihai api_result'taki VLM raporunun ilk 100 karakteri: {api_result['vlm_radiology_report'][:100]}...")


    # Veritabanına kaydetme
    try:
        new_patient = Patient(name=patient_name, surname=patient_surname, age=lab_data_dict.get('Yaş'), gender="Erkek" if lab_data_dict.get('Cinsiyet') == 1 else "Kadın", user_id=user_id)
        db.add(new_patient); db.commit(); db.refresh(new_patient)
        
        patient_details_for_db = {
            "lab_data": lab_data_dict, 
            "afp_value": afp_value, 
            "risk_factors": {
                "alcohol": alcohol_consumption, 
                "smoking": smoking_status,
                "hcv": hcv_status,
                "hbv": hbv_status,
                "cancer_history": cancer_history_status
            }, 
            "usg_report_vlm": vlm_radiology_report # VLM raporunu buraya da ekledik
        } 
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