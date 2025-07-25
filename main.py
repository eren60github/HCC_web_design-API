# hcc_backend_api/main.py

# --- Gerekli Kütüphaneler ---
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional
import json
from datetime import datetime
import pandas as pd
import io
from PIL import Image
import os
import uvicorn
import uuid
import tempfile

# --- Makine Öğrenmesi ve Derin Öğrenme Kütüphaneleri ---
from passlib.context import CryptContext
import joblib
import tensorflow as tf
import numpy as np
import nibabel as nib
from scipy.ndimage import label
import google.generativeai as genai

# --- Yerel Dosyalar ---
# Bu satırların çalışması için projenizde database.py dosyasının olması gerekir.
# Eğer veritabanı kullanmıyorsanız bu satırları ve veritabanı ile ilgili diğer kısımları yorum satırı yapabilirsiniz.
from database import Base, engine, get_db, User, Patient, Evaluation

# --- UYGULAMA AYARLARI ---
# TensorFlow'un bilgi mesajlarını gizle (isteğe bağlı)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Veritabanı ve tabloları oluştur (eğer database.py kullanılıyorsa)
Base.metadata.create_all(bind=engine)

# Şifreleme context'i
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- MODEL ve DOSYA YOLLARI ---
LAB_MODEL_PATH = 'hcc_multi_model_xgboost.joblib'
LAB_SCALER_PATH = 'hcc_scaler_multi.joblib'
USG_MODEL_PATH = 'fibroz_vgg16_model.h5'
MRI_MODEL_PATH = 'MR_model.h5'

# Global model değişkenleri
model_lab, scaler_lab, model_usg, model_mri, vlm_model = None, None, None, None, None

# Sınıf isimleri (Keras USG modeli için)
CLASS_NAMES = ['F0- Fibroz yok', 'F1- Hafif Fibroz', 'F2- Orta Fibroz', 'F3- Ağır Fibroz', 'F4- Siroz']

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

# --- FastAPI UYGULAMASI ve CORS ---
app = FastAPI(title="Gelişmiş HCC Erken Teşhis Sistemi API")

origins = ["http://localhost", "http://localhost:3000"]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# --- MR ANALİZİ İÇİN YARDIMCI FONKSİYONLAR ---
def preprocess_slice_mri(slice_2d, target_size=(128, 128)):
    """MR kesitini modelin beklediği formata getirir."""
    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
    slice_resized = tf.image.resize(slice_2d[..., np.newaxis], target_size)
    return tf.expand_dims(slice_resized, axis=0)

def filter_predicted_volume_mri(volume, min_voxel=500):
    """Tahmin edilen hacimdeki küçük, alakasız alanları filtreler."""
    filtered_volume = volume.copy()
    tumor_mask = (filtered_volume == 2).astype(np.uint8)
    structure = np.ones((3, 3, 3))
    labeled_array, num_features = label(tumor_mask, structure=structure)
    for region_idx in range(1, num_features + 1):
        if np.sum(labeled_array == region_idx) < min_voxel:
            filtered_volume[labeled_array == region_idx] = 0
    return filtered_volume

def count_nodules_mri(volume):
    """Filtrelenmiş hacimdeki ayrı nodül (tümör) sayısını bulur."""
    tumor_mask = (volume == 2).astype(np.uint8)
    structure = np.ones((3, 3, 3))
    _, num_features = label(tumor_mask, structure=structure)
    return num_features

def classify_stage_mri(tumor_ratio, nodule_count):
    """Tümör oranı ve nodül sayısına göre basit bir evreleme yapar."""
    if tumor_ratio > 50 or nodule_count > 5:
        return "Evre 4 - İleri Evre"
    elif tumor_ratio > 25 or nodule_count > 2:
        return "Evre 3 - Orta-İleri Evre"
    elif tumor_ratio > 5 or nodule_count > 0:
        return "Evre 2 - Erken Evre"
    elif tumor_ratio > 0:
        return "Evre 1 - Çok Erken Evre"
    else:
        return "Evre 0 - Tümör Tespit Edilmedi"

# --- MEVCUT YARDIMCI TAHMİN FONKSİYONLARI ---
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
    if model_usg is None: raise HTTPException(status_code=500, detail="USG modeli yüklenmedi.")
    try: 
        image = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize((224, 224))
        image_array = np.array(image) / 255.0
        input_tensor = np.expand_dims(image_array, axis=0)
        predictions = model_usg.predict(input_tensor, verbose=0)
        predicted_class_id = np.argmax(predictions, axis=1)[0]
        return {"stage_label": CLASS_NAMES[predicted_class_id], "stage_id": int(predicted_class_id)}
    except Exception as e: 
        raise HTTPException(status_code=500, detail=f"USG tahmin hatası: {e}. Lütfen geçerli bir görüntü dosyası yüklediğinizden emin olun.")

async def generate_radiology_report_vlm(image_bytes: bytes, predicted_stage_label: str):
    """Görüntü baytları ve tahmin edilen evreye göre VLM'den rapor oluşturur."""
    if vlm_model is None:
        return "VLM raporu oluşturulamadı: Gemini modeli yüklenmedi veya bir hata oluştu."
    
    prompt_template = f"""
# ROL VE BAĞLAM
Sen, uzman bir radyolog tarafından geliştirilen bir yapay zeka modelinin sonuçlarını doğrulayan ve açıklayan bir yapay zeka asistanısın. Bu model, aşağıda sunulan ultrason görüntüsü için **{predicted_stage_label}** sonucunu verdi.

# GÖREV
Senin görevin, bir uzman gibi davranarak bu tahmini doğrulamaktır. Görüntüyü dikkatlice incele ve **sadece {predicted_stage_label}** evresiyle uyumlu olan görsel kanıtları listeleyen, yapılandırılmış bir rapor yaz. Amacın, modelin verdiği kararın GÖRSEL GEREKÇESİNİ oluşturmaktır. Kesinlikle kendi tahminini yapma, sadece verilen tahmini destekleyen kanıtları bul.

# İSTENEN RAPOR FORMATI
Lütfen raporunu aşağıdaki maddelere sadık kalarak oluştur. Her başlığın altına kısa ve net açıklamalar ekle.

**Hasta Görüntüsü Analizi**
---------------------------------
**Ön-Tahmin (Referans Model):** {predicted_stage_label}

**Görsel Bulgular (Tahmini Destekleyen):**
* **Karaciğer Yüzeyi:** [Buraya karaciğer yüzeyinin düzenli mi, nodüler mi, pürüzlü mü olduğunu yaz. Örneğin: "Karaciğer yüzeyi belirgin şekilde nodüler bir görünüm sergilemektedir."]
* **Karaciğer Parankimi (Dokusu):** [Buraya parankimin homojen mi, heterojen mi, kaba mı olduğunu yaz. Örneğin: "Parankim ekosu heterojen ve kaba bir yapıdadır."]
* **Vasküler Yapılar:** [Buraya portal ven ve hepatik venlerin görünürlüğünü ve yapısını yaz. Örneğin: "Vasküler yapılar net olarak ayırt edilememektedir."]
* **Diğer İlgili Bulgular:** [Varsa asit, splenomegali gibi ek bulguları yaz. Örneğin: "Görüntüde asit varlığına dair bir bulgu saptanmamıştır."]

**Sonuç:** [Buraya bulguların, verilen ön-tahminle nasıl uyumlu olduğunu özetleyen kısa bir cümle yaz. Örneğin: "Yukarıdaki görsel bulgular, referans modelin belirttiği **{predicted_stage_label}** tanısıyla uyumludur."]
"""
    try:
        img_for_vlm = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        response = await vlm_model.generate_content_async([prompt_template, img_for_vlm])
        # DÜZELTME: Markdown formatını HTML'e doğru şekilde çevir
        return response.text.replace('\n', '<br>').replace('**', '<b>').replace('</b>', '')
    except Exception as e:
        print(f"VLM Rapor oluşturma hatası: {e}")
        return f"VLM Raporu oluşturma hatası: {e}"

# --- GÜNCELLENMİŞ MR ANALİZİ ANA FONKSİYONU ---
async def predict_mri_analysis(file_bytes: bytes, original_filename: str):
    """Yüklenen .nii dosyasını analiz eder ve tümör oranı, nodül sayısı, evre döndürür."""
    if model_mri is None:
        raise HTTPException(status_code=500, detail="MR modeli yüklenmedi.")

    suffix = ".tmp"
    if original_filename.endswith(".nii.gz"): suffix = ".nii.gz"
    elif original_filename.endswith(".nii"): suffix = ".nii"
    elif original_filename.endswith(".dcm"): suffix = ".dcm"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(file_bytes)
        temp_filepath = tmp.name
    
    try:
        nii = nib.load(temp_filepath)
        volume = nii.get_fdata()
        
        mask_volume_list = []
        for i in range(volume.shape[2]):
            slice_2d = volume[:, :, i]
            input_tensor = preprocess_slice_mri(slice_2d)
            prediction = model_mri.predict(input_tensor, verbose=0)
            prediction_mask = tf.argmax(prediction[0], axis=-1).numpy()
            mask_volume_list.append(prediction_mask)

        mask_volume = np.array(mask_volume_list).transpose((1, 2, 0))
        
        total_liver_pixels = np.sum((mask_volume == 1) | (mask_volume == 2))
        total_tumor_pixels = np.sum(mask_volume == 2)
        tumor_ratio = (total_tumor_pixels / total_liver_pixels) * 100 if total_liver_pixels > 0 else 0.0
        
        filtered_volume = filter_predicted_volume_mri(mask_volume)
        nodule_count = count_nodules_mri(filtered_volume)
        
        if nodule_count == 0 and tumor_ratio > 0: nodule_count = 1
            
        stage = classify_stage_mri(tumor_ratio, nodule_count)

        return {
            "tumor_ratio": round(tumor_ratio, 2),
            "nodule_count": int(nodule_count),
            "stage": stage
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MRI işleme hatası: {e} (Dosya: {original_filename})")
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

# --- API BAŞLANGIÇ OLAYLARI (MODEL YÜKLEME) ---
@app.on_event("startup")
async def load_models_on_startup():
    global model_lab, scaler_lab, model_usg, vlm_model, model_mri
    print("--- Modeller Yükleniyor ---")
    try:
        if os.path.exists(LAB_MODEL_PATH): model_lab = joblib.load(LAB_MODEL_PATH)
        if os.path.exists(LAB_SCALER_PATH): scaler_lab = joblib.load(LAB_SCALER_PATH)
        if os.path.exists(USG_MODEL_PATH): model_usg = tf.keras.models.load_model(USG_MODEL_PATH, compile=False)
        if os.path.exists(MRI_MODEL_PATH): model_mri = tf.keras.models.load_model(MRI_MODEL_PATH, compile=False)
        print(f"✅ Lab Modeli: {'Yüklendi' if model_lab else 'Bulunamadı'}")
        print(f"✅ USG Modeli: {'Yüklendi' if model_usg else 'Bulunamadı'}")
        print(f"✅ MRI Modeli: {'Yüklendi' if model_mri else 'Bulunamadı'}")
        with open('api_key.txt', 'r') as f: GEMINI_API_KEY = f.read().strip()
        genai.configure(api_key=GEMINI_API_KEY)
        vlm_model = genai.GenerativeModel('gemini-1.5-flash-latest')
        print("✅ Gemini VLM Modeli: Yapılandırıldı")
    except Exception as e:
        print(f"HATA: Modeller yüklenirken bir sorun oluştu: {e}")

# --- KULLANICI ENDPOINT'LERİ ---
@app.post("/register", status_code=201)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    db_user = db.query(User).filter(User.email == user.email).first()
    if db_user: raise HTTPException(status_code=400, detail="Bu e-posta adresi zaten kayıtlı.")
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

# --- MERKEZİ TAHMİN ENDPOINT'İ (TAMAMLANMIŞ) ---
@app.post("/evaluate_hcc_risk")
async def evaluate_hcc_risk(
    user_id: int = Form(...),
    patient_name: str = Form(...),
    patient_surname: str = Form(...),
    lab_data: str = Form(...),
    usg_file: Optional[UploadFile] = File(None),
    mri_file: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db),
    afp_value: Optional[float] = Form(None),
    alcohol_consumption: Optional[str] = Form(None),
    smoking_status: Optional[str] = Form(None),
    hcv_status: Optional[str] = Form(None),
    hbv_status: Optional[str] = Form(None),
    cancer_history_status: Optional[str] = Form(None)
):
    overall_risk_level = "Düşük Risk"
    detailed_report_summary = []
    vlm_radiology_report = None
    mri_analysis_result = {}

    try:
        lab_data_dict = json.loads(lab_data)
        lab_data_pydantic = LabData(**lab_data_dict)
        lab_result = await predict_lab_risk(lab_data_pydantic)
        detailed_report_summary.append(f"Laboratuvar Analizi: Tahmin Edilen Hastalık: {lab_result['predicted_disease']}, Risk: {lab_result['risk_level']}")
        overall_risk_level = lab_result['risk_level']
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lab verisi işlenemedi: {e}")

    if usg_file and usg_file.filename:
        try:
            contents = await usg_file.read()
            usg_result = await predict_usg_fibrosis(contents)
            detailed_report_summary.append(f"USG Analizi: Fibrozis Evresi: {usg_result['stage_label']}")
            if usg_result["stage_id"] >= 3: overall_risk_level = "Yüksek Risk"
            elif usg_result["stage_id"] >= 1 and overall_risk_level == "Düşük Risk": overall_risk_level = "Orta Risk"
            vlm_radiology_report = await generate_radiology_report_vlm(contents, usg_result['stage_label'])
        except Exception as e:
            detailed_report_summary.append(f"USG Analizi Hatası: {e}")

    if mri_file and mri_file.filename:
        try:
            mri_contents = await mri_file.read()
            mri_analysis_result = await predict_mri_analysis(mri_contents, mri_file.filename)
            detailed_report_summary.append(f"MRI Analizi: Tahmin Edilen Evre: {mri_analysis_result['stage']}, Tümör Oranı: {mri_analysis_result['tumor_ratio']}%, Nodül Sayısı: {mri_analysis_result['nodule_count']}")
            if mri_analysis_result.get("nodule_count", 0) > 0:
                if overall_risk_level != "Yüksek Risk":
                    overall_risk_level = "Yüksek Risk"
        except Exception as e:
            error_detail = e.detail if isinstance(e, HTTPException) else str(e)
            detailed_report_summary.append(f"MRI Analizi Hatası: {error_detail}")

    # Nihai Öneri Mantığı
    mri_recommendation = overall_risk_level in ["Orta Risk", "Yüksek Risk"]
    if mri_recommendation: final_recommendation = "HCC riski mevcut. Kesin tanı için MRI görüntülemesi ve uzman görüşü ÖNERİLİR."
    elif overall_risk_level == "Yüksek Risk": final_recommendation = "Yüksek düzeyde HCC riski. Uzman değerlendirmesi ve yakın takip önerilir."
    elif overall_risk_level == "Orta Risk": final_recommendation = "Orta düzeyde HCC riski. 6 ayda bir AFP ve USG ile yakın takip önerilir."
    else: final_recommendation = "HCC riski düşük. Rutin yıllık kontroller önerilir."

    # Nihai API sonucunu oluştur
    api_result = {
        "overall_risk_level": overall_risk_level,
        "mri_recommendation": mri_recommendation,
        "final_recommendation": final_recommendation,
        "detailed_report_summary": detailed_report_summary,
        "vlm_radiology_report": vlm_radiology_report,
        "mri_analysis": mri_analysis_result
    }
    
    # Veritabanına kaydetme
    try:
        new_patient = Patient(name=patient_name, surname=patient_surname, age=lab_data_dict.get('Yaş'), gender="Erkek" if lab_data_dict.get('Cinsiyet') == 1 else "Kadın", user_id=user_id)
        db.add(new_patient); db.commit(); db.refresh(new_patient)
        patient_details_for_db = { 
            "lab_data": lab_data_dict, "afp_value": afp_value,
            "risk_factors": {"alcohol": alcohol_consumption, "smoking": smoking_status, "hcv": hcv_status, "hbv": hbv_status, "cancer_history": cancer_history_status},
            "usg_report_vlm": vlm_radiology_report, "mri_analysis_summary": mri_analysis_result
        }
        new_evaluation = Evaluation(patient_id=new_patient.id, evaluation_date=datetime.utcnow(), patient_details_json=json.dumps(patient_details_for_db), api_result_json=json.dumps(api_result))
        db.add(new_evaluation); db.commit()
    except Exception as e:
        db.rollback()
        print(f"Veritabanı Kayıt Hatası: {e}")
        api_result["db_error"] = f"Sonuç üretildi ancak veritabanına kaydedilemedi: {e}"

    return api_result

# --- API'yi Çalıştırma ---
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
