# hcc_backend_api/main.py

# --- Gerekli Kütüphaneler ---
from fastapi import FastAPI, HTTPException, Depends, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from pydantic import BaseModel
from typing import Optional, Dict
import json
from datetime import datetime
import pandas as pd
import io
from PIL import Image
import os
import uvicorn
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
from database import Base, engine, get_db, User, Patient, Evaluation 

# --- UYGULAMA AYARLARI ---
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 
Base.metadata.create_all(bind=engine) 
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# --- MODEL ve DOSYA YOLLARI ---
LAB_MODEL_PATH = 'hcc_multi_model_xgboost.joblib'
LAB_SCALER_PATH = 'hcc_scaler_multi.joblib'
USG_MODEL_PATH = 'fibroz_vgg16_model.h5'
MRI_MODEL_PATH = 'MR_model.h5'

# Global model değişkenleri
model_lab, scaler_lab, model_usg, model_mri, gemini_model = None, None, None, None, None

CLASS_NAMES = ['F0- Fibroz yok', 'F1- Hafif Fibroz', 'F2- Orta Fibroz', 'F3- Ağır Fibroz', 'F4- Siroz'] 
# --- PYDANTIC MODELLERİ ---
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


# --- YARDIMCI FONKSİYONLAR ---
def preprocess_slice_mri(slice_2d, target_size=(128, 128)): 
    slice_2d = (slice_2d - np.min(slice_2d)) / (np.max(slice_2d) - np.min(slice_2d) + 1e-8)
    slice_resized = tf.image.resize(slice_2d[..., np.newaxis], target_size) 
    return tf.expand_dims(slice_resized, axis=0) 
def filter_predicted_volume_mri(volume, min_voxel=500): 
    filtered_volume = volume.copy()
    tumor_mask = (filtered_volume == 2).astype(np.uint8)
    structure = np.ones((3, 3, 3))
    labeled_array, num_features = label(tumor_mask, structure=structure)
    for region_idx in range(1, num_features + 1):
        if np.sum(labeled_array == region_idx) < min_voxel:
            filtered_volume[labeled_array == region_idx] = 0
    return filtered_volume
def count_nodules_mri(volume): 
    tumor_mask = (volume == 2).astype(np.uint8)
    structure = np.ones((3, 3, 3))
    _, num_features = label(tumor_mask, structure=structure)
    return num_features
def classify_stage_mri(tumor_ratio, nodule_count):
    if tumor_ratio > 50 or nodule_count > 5: return "Evre 4 - İleri Evre"
    elif tumor_ratio > 25 or nodule_count > 2: return "Evre 3 - Orta-İleri Evre"
    elif tumor_ratio > 5 or nodule_count > 0: return "Evre 2 - Erken Evre"
    elif tumor_ratio > 0: return "Evre 1 - Çok Erken Evre"
    else: return "Evre 0 - Tümör Tespit Edilmedi"
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
    image = Image.open(io.BytesIO(file_bytes)).convert('RGB').resize((224, 224))
    image_array = np.array(image) / 255.0
    input_tensor = np.expand_dims(image_array, axis=0)
    predictions = model_usg.predict(input_tensor, verbose=0)
    predicted_class_id = np.argmax(predictions, axis=1)[0]
    return {"stage_label": CLASS_NAMES[predicted_class_id], "stage_id": int(predicted_class_id)}
async def predict_mri_analysis(file_bytes: bytes, original_filename: str):
    if model_mri is None: raise HTTPException(status_code=500, detail="MR modeli yüklenmedi.")
    suffix = ".tmp"
    if original_filename.endswith((".nii.gz", ".nii", ".dcm")): suffix = os.path.splitext(original_filename)[1]
    if original_filename.endswith(".nii.gz"): suffix = ".nii.gz"
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
        return {"tumor_ratio": round(tumor_ratio, 2), "nodule_count": int(nodule_count), "stage": stage}
    finally:
        if os.path.exists(temp_filepath):
            os.remove(temp_filepath)

# --- AYRI GEMINI FONKSİYONLARI ---
# 1. Görüntü Analizi (VLM) için Gemini Fonksiyonu
async def generate_radiology_report_vlm(image_bytes: bytes, predicted_stage_label: str):
    if gemini_model is None: return "VLM raporu oluşturulamadı: Gemini modeli yüklenmedi."
    prompt_template = f"Bir uzman radyolog gibi davranarak, sana verdiğim bu ultrason görüntüsünü incele ve `{predicted_stage_label}` tanısını destekleyen görsel kanıtları listeleyen, yapılandırılmış bir rapor yaz."
    try:
        img_for_vlm = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        response = await gemini_model.generate_content_async([prompt_template, img_for_vlm])
        return response.text.replace('\n', '<br>').replace('**', '<b>').replace('</b>', '')
    except Exception as e:
        return f"Gemini VLM Raporu oluşturma hatası: {e}"

# 2. Bütünsel Metin Raporu (LLM) için Gemini Fonksiyonu
async def generate_gemini_comprehensive_report(context: Dict):
    if gemini_model is None: return "Bütünsel rapor oluşturulamadı: Gemini modeli yüklenmemiş."
    lab_report = context.get("lab_result", {})
    usg_report = context.get("usg_result", {})
    mri_report = context.get("mri_analysis", {})
    patient_info = context.get("patient_details", {})
    prompt = f"""
# GÖREV VE ROL
Sen, hepatoloji ve onkoloji alanlarında uzmanlaşmış, farklı tıbbi verileri sentezleyerek kapsamlı bir klinik değerlendirme raporu hazırlayan bir yapay zeka asistanısın. Görevin, aşağıda sunulan verileri analiz ederek bütüncül, yapılandırılmış ve profesyonel bir tıbbi rapor oluşturmaktır.
# HASTA VERİLERİ
---------------------------------
**Demografik Bilgiler:**
* **Yaş:** {patient_info.get('age', 'Belirtilmemiş')}
* **Cinsiyet:** {patient_info.get('gender', 'Belirtilmemiş')}
**Risk Faktörleri:**
* **Alkol Tüketimi:** {patient_info.get('alcohol_consumption', 'Belirtilmemiş')}
* **Sigara Kullanımı:** {patient_info.get('smoking_status', 'Belirtilmemiş')}
* **HCV Durumu:** {patient_info.get('hcv_status', 'Belirtilmemiş')}
* **HBV Durumu:** {patient_info.get('hbv_status', 'Belirtilmemiş')}
* **Ailede Kanser Öyküsü:** {patient_info.get('cancer_history_status', 'Belirtilmemiş')}
**ANALİZ SONUÇLARI**
---------------------------------
**1. Laboratuvar Veri Analizi (XGBoost Model):**
* **Tahmin Edilen Durum:** {lab_report.get('predicted_disease', 'Hesaplanmadı')}
* **HCC Olasılığı:** %{lab_report.get('hcc_probability', 0) * 100:.2f}
* **Laboratuvar Bazlı Risk:** {lab_report.get('risk_level', 'Hesaplanmadı')}
**2. Ultrason (USG) Görüntü Analizi (VGG16 Model):**
* **Tespit Edilen Fibrozis Evresi:** {usg_report.get('stage_label', 'USG verisi yok')}
**3. Manyetik Rezonans (MR) Görüntü Analizi (3D U-Net Model):**
* **Tespit Edilen Nodül Sayısı:** {mri_report.get('nodule_count', 'MR verisi yok')}
* **Karaciğerdeki Tümör Oranı:** %{mri_report.get('tumor_ratio', 'MR verisi yok')}
* **MR Bazlı Evre Tahmini:** {mri_report.get('stage', 'MR verisi yok')}
# İSTENEN RAPOR FORMATI
Lütfen aşağıdaki başlıkları kullanarak, yukarıdaki verileri sentezleyen detaylı bir rapor oluştur.
**1. Klinik Özet:**
(Hastanın genel durumu ve en önemli bulgular hakkında 1-2 cümlelik bir özet.)
**2. Bulguların Entegrasyonu ve Yorumlanması:**
(Laboratuvar, USG ve MR bulgularının birbiriyle nasıl ilişkili olduğunu açıkla.)
**3. Bütünsel Risk Değerlendirmesi:**
(Tüm veriler ışığında hastanın genel HCC riskini belirt ve bu sonuca nasıl ulaştığını açıkla.)
**4. Klinik Öneri ve Sonraki Adımlar:**
(Bu bulgular doğrultusunda hekime yönelik spesifik öneriler sun.)
"""
    try:
        response = await gemini_model.generate_content_async(prompt)
        return response.text
    except Exception as e:
        return f"Gemini Bütünsel Rapor oluşturma hatası: {e}"

@app.on_event("startup")
async def load_models_on_startup():
    global model_lab, scaler_lab, model_usg, model_mri, gemini_model
    print("--- Modeller Yükleniyor ---")
    try:
        if os.path.exists(LAB_MODEL_PATH): model_lab = joblib.load(LAB_MODEL_PATH)
        if os.path.exists(LAB_SCALER_PATH): scaler_lab = joblib.load(LAB_SCALER_PATH)
        if os.path.exists(USG_MODEL_PATH): model_usg = tf.keras.models.load_model(USG_MODEL_PATH, compile=False)
        if os.path.exists(MRI_MODEL_PATH): model_mri = tf.keras.models.load_model(MRI_MODEL_PATH, compile=False)
        print(f"✅ Lab, USG, MRI Modelleri: Yüklendi")

        if os.path.exists('api_key.txt'):
            with open('api_key.txt', 'r') as f: GEMINI_API_KEY = f.read().strip()
            genai.configure(api_key=GEMINI_API_KEY)
            gemini_model = genai.GenerativeModel('gemini-1.5-flash-latest')
            print("✅ Gemini Modeli (VLM ve LLM için): Yapılandırıldı")
        else:
            print("⚠️ UYARI: `api_key.txt` (Gemini için) dosyası bulunama   dı.")
    except Exception as e:
        print(f"HATA: Modeller yüklenirken bir sorun oluştu: {e}")

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
    overall_risk_level, detailed_report_summary, vlm_radiology_report = "Düşük Risk", [], None
    mri_analysis_result, lab_result, usg_result, gemini_comprehensive_report = {}, {}, {}, None

    try:
        lab_data_dict = json.loads(lab_data)
        lab_data_pydantic = LabData(**lab_data_dict)
        lab_result = await predict_lab_risk(lab_data_pydantic)
        detailed_report_summary.append(f"Laboratuvar Analizi: {lab_result['predicted_disease']}, Risk: {lab_result['risk_level']}")
        overall_risk_level = lab_result['risk_level']
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Lab verisi hatası: {e}")

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
            detailed_report_summary.append(f"MRI Analizi: Evre: {mri_analysis_result['stage']}, Oran: {mri_analysis_result['tumor_ratio']}%, Nodül: {mri_analysis_result['nodule_count']}")
            if mri_analysis_result.get("nodule_count", 0) > 0: overall_risk_level = "Yüksek Risk"
        except Exception as e:
            detailed_report_summary.append(f"MRI Analizi Hatası: {e.detail if isinstance(e, HTTPException) else str(e)}")

    comprehensive_context = {
        "patient_details": {"age": lab_data_dict.get('Yaş'), "gender": "Erkek" if lab_data_dict.get('Cinsiyet') == 1 else "Kadın", "alcohol_consumption": alcohol_consumption, "smoking_status": smoking_status, "hcv_status": hcv_status, "hbv_status": hbv_status, "cancer_history_status": cancer_history_status, "afp_value": afp_value},
        "lab_result": lab_result, "usg_result": usg_result, "mri_analysis": mri_analysis_result
    }
    gemini_comprehensive_report = await generate_gemini_comprehensive_report(comprehensive_context)
    detailed_report_summary.append("Gemini Bütünsel Değerlendirmesi: Başarıyla oluşturuldu.")

    if overall_risk_level == "Yüksek Risk": final_recommendation = "Yüksek düzeyde HCC riski. Uzman değerlendirmesi, biyopsi ve multidisipliner konsey önerilir."
    elif overall_risk_level == "Orta Risk": final_recommendation = "Orta düzeyde HCC riski. MRI görüntülemesi ve 6 ayda bir AFP/USG ile yakın takip önerilir."
    else: final_recommendation = "HCC riski düşük. Rutin yıllık kontroller önerilir."
    
    api_result = {
        "overall_risk_level": overall_risk_level,
        "final_recommendation": final_recommendation, 
        "detailed_report_summary": detailed_report_summary,
        "vlm_radiology_report": vlm_radiology_report,
        "mri_analysis": mri_analysis_result,
        "gemini_comprehensive_report": gemini_comprehensive_report
    }

    try:
        new_patient = Patient(name=patient_name, surname=patient_surname, age=lab_data_dict.get('Yaş'), gender="Erkek" if lab_data_dict.get('Cinsiyet') == 1 else "Kadın", user_id=user_id)
        db.add(new_patient); db.commit(); db.refresh(new_patient)
        patient_details_for_db = {
            "lab_data": lab_data_dict, "afp_value": afp_value,
            "risk_factors": {"alcohol": alcohol_consumption, "smoking": smoking_status, "hcv": hcv_status, "hbv": hbv_status, "cancer_history": cancer_history_status},
            "usg_report_vlm": vlm_radiology_report, "mri_analysis_summary": mri_analysis_result,
            "gemini_report": gemini_comprehensive_report
        }
        new_evaluation = Evaluation(patient_id=new_patient.id, evaluation_date=datetime.utcnow(), patient_details_json=json.dumps(patient_details_for_db), api_result_json=json.dumps(api_result))
        db.add(new_evaluation); db.commit()
    except Exception as e:
        db.rollback()
        print(f"Veritabanı Kayıt Hatası: {e}")
        api_result["db_error"] = f"Sonuç üretildi ancak veritabanına kaydedilemedi: {e}"

    return api_result

if __name__ == "__main__": 
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True) 