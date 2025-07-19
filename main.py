# main.py içeriği (Birleşik API: Lab + USG + Merkezi Karar)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
import pandas as pd
import joblib
import numpy as np
import os
import uvicorn
import io
from PIL import Image
import tensorflow as tf
import cv2
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

# --- AYARLAR ---
LAB_MODEL_PATH = 'ensemble_model.joblib'
USG_MODEL_PATH = 'fibroz_vgg16_model.h5'
# MRI_MODEL_PATH = 'mri_model.h5' # MRI modeliniz hazır olduğunda burayı ve yüklemesini ekleyeceğiz.



model_lab = None
model_usg = None
# model_mri = None # MRI modeliniz hazır olduğunda burayı ekleyeceğiz.

app = FastAPI(
    title="HCC Erken Teşhis Sistemi API",
    description="HCC risk tahmini için laboratuvar, USG ve MRI verilerini işleyen API'ler."
)

# --- CORS AYARLARI ---
origins = [
    "http://localhost",
    "http://localhost:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- API Başlamadan Önce Modelleri Yükleme ---
@app.on_event("startup")
async def load_models():
    global model_lab, model_usg # , model_mri # MRI modeliniz hazır olduğunda burayı ekleyeceğiz.
    try:
        if not os.path.exists(LAB_MODEL_PATH):
            raise FileNotFoundError(f"Lab modeli dosyası bulunamadı: {LAB_MODEL_PATH}")
        model_lab = joblib.load(LAB_MODEL_PATH)
        print(f"Lab modeli başarıyla yüklendi: {LAB_MODEL_PATH}")

        if not os.path.exists(USG_MODEL_PATH):
            raise FileNotFoundError(f"USG modeli dosyası bulunamadı: {USG_MODEL_PATH}")
        model_usg = tf.keras.models.load_model(USG_MODEL_PATH, compile=False)
        print(f"USG modeli başarıyla yüklendi: {USG_MODEL_PATH}")

        # # MRI modelini yükle (MRI modeli hazır olduğunda uncomment edin)
        # if not os.path.exists(MRI_MODEL_PATH):
        #     raise FileNotFoundError(f"MRI modeli dosyası bulunamadı: {MRI_MODEL_PATH}")
        # model_mri = tf.keras.models.load_model(MRI_MODEL_PATH, compile=False)
        # print(f"MRI modeli başarıyla yüklendi: {MRI_MODEL_PATH}")

    except Exception as e:
        print(f"HATA: Modeller yüklenirken bir sorun oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"Modeller yüklenemedi: {e}")

# --- Lab Veri Modeli Tanımlama ---
class LabData(BaseModel):
    Age: float
    Gender: int
    Total_Bilirubin: float
    Direct_Bilirubin: float
    Alkaline_Phosphotase: float
    Alamine_Aminotransferase: float
    Aspartate_Aminotransferase: float
    Total_Protiens: float
    Albumin: float
    Albumin_and_Globulin_Ratio: float

# --- Lab Modeli API Endpoint'i ---
# Bu endpoint, merkezi endpoint tarafından çağrılacağı için burada kalacak.
@app.post("/predict_lab_risk")
async def predict_lab_risk(data: LabData):
    """
    Hasta laboratuvar değerlerine göre karaciğer hastalığı riskini tahmin eder.
    Bu endpoint, tekil Lab modeli tahminini döndürür.
    """
    if model_lab is None:
        raise HTTPException(status_code=500, detail="Lab modeli henüz yüklenmedi.")

    features_order = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 'Alkaline_Phosphotase',
        'Alamine_Aminotransferase', 'Aspartate_Aminotransferase', 'Total_Protiens',
        'Albumin', 'Albumin_and_Globulin_Ratio'
    ]
    input_df = pd.DataFrame([data.model_dump()], columns=features_order)

    prediction_proba = model_lab.predict_proba(input_df)[0].tolist()
    prediction_raw = model_lab.predict(input_df)[0]

    risk_level = "Belirlenemedi"
    pred_prob = prediction_proba[1] # Hasta olma olasılığı

    if pred_prob <= 0.30:
        risk_level = "Düşük Risk (Karaciğer Hastalığı Belirtisi Yok)"
    elif pred_prob <= 0.62:
        risk_level = "Orta Risk (Hafif Karaciğer Hastalığı Şüphesi)"
    else:
        risk_level = "Yüksek Risk (Belirgin Karaciğer Hastalığı Şüphesi)"

    return {
        "status": "success",
        "predicted_class_id": int(prediction_raw),
        "predicted_class_label": "Liver Patient" if prediction_raw == 1 else "Non-Liver Patient",
        "prediction_probabilities": {"Non-Liver Patient": prediction_proba[0], "Liver Patient": prediction_proba[1]},
        "risk_level": risk_level,
        "message": "Lab verilerine göre karaciğer hastalığı riski tahmini."
    }

# --- USG Modeli API Endpoint'i ---
# Bu endpoint, merkezi endpoint tarafından çağrılacağı için burada kalacak.
@app.post("/predict_usg_fibrosis")
async def predict_usg_fibrosis(file: UploadFile = File(...)):
    """
    USG görüntüsü yükleyerek karaciğer fibrozis evresini (F0-F4) tahmin eder.
    Bu endpoint, tekil USG modeli tahminini döndürür.
    """
    if model_usg is None:
        raise HTTPException(status_code=500, detail="USG modeli henüz yüklenmedi.")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('L') # Gri tonlamalı
        image = image.resize((224, 224)) # VGG16 için 224x224
        image_array = np.array(image)

        image_array = np.expand_dims(image_array, axis=-1)
        image_array_rgb = np.repeat(image_array, 3, axis=-1)
        image_array_rgb = image_array_rgb / 255.0
        input_tensor = np.expand_dims(image_array_rgb, axis=0)

        predictions = model_usg.predict(input_tensor)
        predicted_class_id = np.argmax(predictions, axis=1)[0]
        prediction_probabilities = predictions[0].tolist()

        fibrosis_stages_labels = ['F0- Fibroz yok', 'F1- Hafif Fibroz', 'F2- Orta Fibroz', 'F3- Ağır Fibroz', 'F4- Siroz']
        predicted_stage_label = fibrosis_stages_labels[predicted_class_id]

        proba_dict = {
            stage: float(prob) for stage, prob in zip(fibrosis_stages_labels, prediction_probabilities)
        }

        return {
            "status": "success",
            "predicted_fibrosis_stage_id": int(predicted_class_id),
            "predicted_fibrosis_stage_label": predicted_stage_label,
            "prediction_probabilities": proba_dict,
            "message": "USG görüntüsüne göre karaciğer fibrozis evresi tahmini."
        }

    except Exception as e:
        print(f"HATA: USG görüntüsü işlenirken veya tahmin yapılırken bir sorun oluştu: {e}")
        raise HTTPException(status_code=500, detail=f"USG görüntüsü işlenemedi veya tahmin yapılamadı: {e}")

# --- Merkezi Karar Verme Endpoint'i (Agentic AI Çekirdeği) ---
@app.post("/evaluate_hcc_risk")
async def evaluate_hcc_risk(
    lab_data: str = Form(..., description="JSON formatında Lab verileri (Age, Gender, Total_Bilirubin, vb.)"),
    # usg_file ve mri_file'ı Optional yaparak ve default'u None ile vererek
    # hiçbir dosya yüklenmediğinde hata vermemesini sağlıyoruz.
    usg_file: Optional[UploadFile] = File(None, description="İsteğe bağlı USG Görüntüsü Dosyası (JPG/PNG)"),
    mri_file: Optional[UploadFile] = File(None, description="İsteğe bağlı MRI Görüntüsü Dosyası (NIfTI veya diğer)")
):
    is_usg_file_present = False
    is_mri_file_present = False

    """
    Hasta laboratuvar verileri ve isteğe bağlı görüntüleme verilerini (USG/MRI) alarak
    birleşik HCC riski değerlendirmesi ve takip/tedavi önerisi sunar.
    Bu, projenizin Agentic AI çekirdeğidir.
    """
    overall_risk_level = "Belirlenemedi"
    detailed_report_summary = [] # LLM'e gidecek detaylı rapor özeti. Burası doğru isim.
    mri_recommendation = False
    final_recommendation = "Genel Karaciğer Sağlığı İyi. Rutin Kontrollere Devam."

    print(f"\n--- Yeni İstek: /evaluate_hcc_risk ---")
    print(f"Alınan Lab Verisi (string): {lab_data}")
    print(f"USG Dosyası Var mı?: {usg_file is not None}")
    print(f"MRI Dosyası Var mı?: {mri_file is not None}")

    # 1. Lab Modelini Çalıştır
    lab_risk_result = None
    try:
        lab_data_parsed = LabData.model_validate_json(lab_data)
        lab_prediction_response = await predict_lab_risk(lab_data_parsed)
        lab_risk_result = {
            "level": lab_prediction_response["risk_level"],
            "proba_liver_patient": lab_prediction_response["prediction_probabilities"]["Liver Patient"]
        }
        detailed_report_summary.append(f"Laboratuvar Analizi: Risk Seviyesi: {lab_risk_result['level']} (Hasta Olasılığı: %{round(lab_risk_result['proba_liver_patient'] * 100, 2)})")

        if "Yüksek Risk" in lab_risk_result["level"]:
            overall_risk_level = "Yüksek Risk"
            mri_recommendation = True
            detailed_report_summary.append("Lab verileri, belirgin karaciğer hastalığı şüphesi nedeniyle ileri görüntüleme (MRI) gerektirebilir.")
        elif "Orta Risk" in lab_risk_result["level"]:
            overall_risk_level = "Orta Risk"
            # Hata burada detailed_message yerine detailed_report_summary olmalıydı
            detailed_report_summary.append("Lab verileri, hafif karaciğer hastalığı şüphesi taşımaktadır.")
        else:
            overall_risk_level = "Düşük Risk"
            detailed_report_summary.append("Lab verileri, karaciğer hastalığı belirtisi göstermemektedir.")

    except Exception as e:
        detailed_report_summary.append(f"HATA: Laboratuvar verileri işlenirken bir sorun oluştu: {e}")
        print(f"HATA: Lab verileri işlenirken: {e}")
        lab_risk_result = {"level": "Hata", "proba_liver_patient": 0}


    # 2. USG Görüntüsü Varsa USG Modelini Çalıştır
    usg_fibrosis_result = None
    if usg_file:
        try:
            usg_prediction_response = await predict_usg_fibrosis(usg_file)
            usg_fibrosis_result = {
                "stage_label": usg_prediction_response["predicted_fibrosis_stage_label"],
                "stage_id": usg_prediction_response["predicted_fibrosis_stage_id"]
            }
            detailed_report_summary.append(f"USG Görüntü Analizi: Karaciğer Fibrozis Evresi: {usg_fibrosis_result['stage_label']}")

            if usg_fibrosis_result["stage_id"] >= 3:
                if overall_risk_level != "Yüksek Risk":
                    overall_risk_level = "Yüksek Risk"
                mri_recommendation = True
                detailed_report_summary.append(f"USG bulguları ({usg_fibrosis_result['stage_label']}), HCC riski için belirgin bir faktör olup ileri görüntüleme (MRI) gerektirebilir.")
            elif usg_fibrosis_result["stage_id"] >= 1:
                if overall_risk_level == "Düşük Risk":
                    overall_risk_level = "Orta Risk"
                detailed_report_summary.append(f"USG bulguları ({usg_fibrosis_result['stage_label']}), karaciğerde fibrozis belirtileri göstermektedir. Yakın takip önerilir.")
            else:
                 detailed_report_summary.append("USG bulguları (F0-Fibroz yok), karaciğerde fibrozis belirtisi göstermemektedir.")

        except Exception as e:
            detailed_report_summary.append(f"HATA: USG görüntüsü işlenirken bir sorun oluştu: {e}")
            print(f"HATA: USG görüntüsü işlenirken: {e}")
            usg_fibrosis_result = None


    # 3. MRI Görüntüsü Varsa MRI Modelini Çalıştır (Placeholder - Henüz Model Yok)
    mri_tumor_result = None
    if mri_file:
        detailed_report_summary.append("MRI Görüntüsü yüklendi.")
        # Burada MRI modeli çalıştırılacak ve sonuçları eklenecek
        # Şimdilik, MRI yüklendiği için riski yükseltelim ve MRI'ı zaten çekildiği için öneriyi kaldırabiliriz.
        mri_recommendation = False # MRI yüklendiği için artık MRI önerisi yapmıyoruz
        overall_risk_level = "Yüksek Risk" # MRI yüklüyse risk yüksek kabul edilir
        detailed_report_summary.append("MRI analizi (tümör boyutu, HCC evresi) burada yapılacaktır.")

        # # Gerçek MRI modeli entegrasyonu (model_mri hazır olduğunda)
        # try:
        #     # mri_prediction_response = await predict_mri_hcc(mri_file) # İleride yazılacak
        #     # mri_tumor_result = {
        #     #     "tumor_size": mri_prediction_response["tumor_size"],
        #     #     "hcc_stage": mri_prediction_response["hcc_stage"]
        #     # }
        #     # detailed_report_summary.append(f"MRI Analizi: Tespit Edilen Tümör Boyutu: {mri_tumor_result['tumor_size']}, HCC Evresi: {mri_tumor_result['hcc_stage']}")
        #     # overall_risk_level = "Yüksek Risk"
        #     # final_recommendation = "HCC tanısı konmuştur. Uzman onkolog ile tedavi planlaması önerilir."
        # except Exception as e:
        #     detailed_report_summary.append(f"HATA: MRI görüntüsü işlenirken bir sorun oluştu: {e}")
        #     print(f"HATA: MRI görüntüsü işlenirken: {e}")


    # 4. Nihai Öneri (Agentic AI Kararı)
    # Bu kısım, dokümanınızdaki 'Tedavi önerisi' ve 'Takip önerileri'ne göre şekillenecektir.
    if is_mri_file_present: # Eğer MRI dosyası yüklendiyse (yani MRI çekilmişse)
        if mri_tumor_result: # Ve MRI modeli de çalışıp tümör bulduysa (ileride)
            final_recommendation = "HCC tanısı ve evrelemesi yapıldı. Uzman onkolog ile tedavi planlaması önerilir."
        else: # MRI yüklendi ama henüz modeli yok veya tümör analizi sonucu gelmedi (şimdiki durum)
            final_recommendation = "MRI görüntülemesi yapıldı. Detaylı analiz (tümör boyutu, HCC evresi) bekleniyor. Uzman değerlendirmesi önemlidir."
        # Overall risk zaten yukarıda "Yüksek Risk" olarak ayarlanıyor mri_file varsa
    elif mri_recommendation: # Eğer MRI önerisi Lab veya USG'den geldiyse (MRI henüz çekilmediyse)
        final_recommendation = "HCC riski yüksek. Kesin tanı ve evreleme için MRI görüntülemesi ŞİDDETLE ÖNERİLİR."
        overall_risk_level = "Yüksek Risk" # Eğer buraya düşüyorsa zaten yüksek risk vardır
    elif overall_risk_level == "Yüksek Risk": # Sadece Lab veya USG'den yüksek risk gelmiş ve MRI önerisi yoksa (nadiren)
        final_recommendation = "Yüksek düzeyde HCC riski. Uzman gastroenterolog/hepatolog değerlendirmesi ve yakın takip (3 ayda bir AFP ve USG) önerilir."
    elif overall_risk_level == "Orta Risk":
        final_recommendation = "Orta düzeyde HCC riski. 6 ayda bir AFP ve USG ile yakın takip önerilir. Uzman gastroenterolog/hepatolog değerlendirmesi düşünülebilir."
    else: # overall_risk_level == "Düşük Risk" ise
        final_recommendation = "HCC riski düşük. Rutin yıllık kontroller (Lab testleri ve USG) önerilir."

    # Not: AFP değeri şu an LabData modelinde yok, ancak olsaydı buraya ekleyebilirdik.


    return {
        "status": "success",
        "overall_risk_level": overall_risk_level,
        "mri_recommendation": mri_recommendation,
        "final_recommendation": final_recommendation,
        "detailed_report_summary": detailed_report_summary
    }



# --- API'yi Yerelde Çalıştırma ---
if __name__ == "__main__":
    print(f"\n--- Birleşik Backend API'si Başlatılıyor (Yerel Port: 8000) ---")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    print("\n-------------------------------------------------------------")
    print("BİRLEŞİK BACKEND API'Sİ ÇALIŞIYOR. http://localhost:8000/docs adresinden test edebilirsiniz.")
    print("---------------------------------------------------------------------------------------------------")