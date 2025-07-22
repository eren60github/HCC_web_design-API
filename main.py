# hcc_backend_api/main.py (FİNAL GÜNCEL VERSİYON: Yeni Lab ve USG Modelleri Entegre Edildi)
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
import os
import uvicorn
import io
from PIL import Image
import tensorflow as tf
import cv2
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, Union # Optional ve Union ekledik
import json # JSON string'i parse etmek için eklendi

# --- AYARLAR ---
# Model ve scaler dosya yolları (Lütfen kendi kaydettiğiniz dosya adları ve yollarıyla eşleştirin)
LAB_MODEL_PATH = 'hcc_multi_model_xgboost.joblib' # Yeni Lab modelinizin adı
LAB_SCALER_PATH = 'hcc_scaler_multi.joblib'     # Yeni Lab scaler'ınızın adı
USG_MODEL_PATH = 'fibroz_vgg16_model.h5'      # USG modelinizin adı
# MRI_MODEL_PATH = 'mri_model.h5' # MRI modeliniz hazır olduğunda burayı ve yüklemesini ekleyeceğiz.

model_lab = None
scaler_lab = None # Yeni Lab modeli için scaler
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
    # Frontend'iniz farklı bir portta (genellikle 3000) çalıştığı için CORS'a izin vermemiz gerekir.
    # Üretim ortamında buraya web sitenizin gerçek URL'sini ekleyin.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], # GET, POST, PUT, DELETE vb. tüm metotlara izin ver
    allow_headers=["*"], # Tüm başlıklara izin ver
)

# --- API Başlamadan Önce Modelleri Yükleme ---
@app.on_event("startup")
async def load_models():
    global model_lab, scaler_lab, model_usg # , model_mri
    try:
        # Yeni Lab modelini ve scaler'ı yükle
        if not os.path.exists(LAB_MODEL_PATH):
            raise FileNotFoundError(f"Lab modeli dosyası bulunamadı: {LAB_MODEL_PATH}")
        if not os.path.exists(LAB_SCALER_PATH):
            raise FileNotFoundError(f"Lab scaler dosyası bulunamadı: {LAB_SCALER_PATH}")

        model_lab = joblib.load(LAB_MODEL_PATH)
        scaler_lab = joblib.load(LAB_SCALER_PATH)
        print(f"Yeni Lab modeli ({LAB_MODEL_PATH}) ve scaler ({LAB_SCALER_PATH}) başarıyla yüklendi.")

        # USG modelini yükle
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

# --- Yeni Lab Veri Modeli Tanımlama (Yeni Özellik İsimleriyle) ---
# Bu model, Lab modelinizin beklediği EXACT özellik isimlerini içermelidir.
class LabData(BaseModel):
    Yaş: float
    Cinsiyet: int # 0: Kadın, 1: Erkek (LabelEncoder'dan geliyor)
    Albumin: float
    ALP: float # Alkaline Phosphotase
    ALT: float # Alamine Aminotransferase
    AST: float # Aspartate Aminotransferase
    BIL: float # Total Bilirubin
    GGT: float

    # Not: Direct_Bilirubin, Total_Protiens, Albumin_and_Globulin_Ratio artık Lab modelinin direk girdisi değil.
    # Bunlar LabData modelinden kaldırıldı.


# --- Lab Modeli API Endpoint'i ---
@app.post("/predict_lab_risk")
async def predict_lab_risk(data: LabData):
    """
    Hasta laboratuvar değerlerine göre karaciğer hastalığı riskini ve tahmin edilen hastalığı tahmin eder.
    Bu endpoint, tekil Lab modeli tahminini döndürür.
    """
    if model_lab is None or scaler_lab is None:
        raise HTTPException(status_code=500, detail="Lab modeli veya scaler henüz yüklenmedi.")

    # Yeni özellik sırası (HCC_FinalTest.ipynb'deki X.columns.tolist() çıktısıyla aynı olmalı)
    features_order_lab = ['Yaş', 'Cinsiyet', 'Albumin', 'ALP', 'ALT', 'AST', 'BIL', 'GGT']

    # Gelen veriyi Pandas DataFrame'e dönüştür (tek bir örnek için)
    # Pydantic model names (Yaş, Cinsiyet, vb.) doğrudan DataFrame sütun adları olacak.
    input_df_lab = pd.DataFrame([data.model_dump()], columns=features_order_lab)

    # Veriyi ölçeklendir (modelin eğitildiği scaler ile)
    # Yeni Lab modeli ölçeklenmiş veri bekler.
    input_df_lab_scaled = scaler_lab.transform(input_df_lab)

    # Tahmin yap
    predictions_proba_all_classes = model_lab.predict_proba(input_df_lab_scaled)[0]
    predicted_class_id_lab = np.argmax(predictions_proba_all_classes)

    # HCC_FinalTest.ipynb'deki hastalık isimleri
    disease_names_map = {0: "Sağlıklı", 1: "Hepatit", 2: "Fibröz", 3: "Siroz", 4: "HCC"}
    predicted_disease_label_lab = disease_names_map.get(predicted_class_id_lab, "Bilinmiyor")

    # HCC olasılığı (genellikle 4. indeks)
    hcc_probability = predictions_proba_all_classes[4] if len(predictions_proba_all_classes) > 4 else 0.0

    # Risk seviyelerini belirle (HCC_FinalTest.ipynb'deki yeni eşikler: 0.33, 0.66)
    lab_risk_level = "Belirlenemedi"
    if hcc_probability < 0.33:
        lab_risk_level = "Düşük Risk (Karaciğer Hastalığı Belirtisi Yok)"
    elif hcc_probability < 0.66:
        lab_risk_level = "Orta Risk (Hafif Karaciğer Hastalığı Şüphesi)"
    else:
        lab_risk_level = "Yüksek Risk (Belirgin Karaciğer Hastalığı Şüphesi)"

    # Tüm sınıf olasılıklarını etiketleriyle birlikte sözlüğe dönüştür
    all_class_probabilities_dict = {
        disease_names_map.get(i, f"Sınıf {i}"): float(prob)
        for i, prob in enumerate(predictions_proba_all_classes)
    }

    return {
        "status": "success",
        "predicted_disease_id": int(predicted_class_id_lab),
        "predicted_disease_label": predicted_disease_label_lab,
        "hcc_probability": float(hcc_probability), # Sadece HCC sınıfının olasılığı
        "all_class_probabilities": all_class_probabilities_dict, # Tüm sınıfların olasılıkları
        "risk_level": lab_risk_level, # Düşük, Orta, Yüksek risk seviyesi
        "message": "Lab verilerine göre karaciğer hastalığı ve HCC riski tahmini."
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

        image_array = np.expand_dims(image_array, axis=-1) # (224, 224, 1)
        image_array_rgb = np.repeat(image_array, 3, axis=-1) # (224, 224, 3)

        # Normalizasyon (0-255 -> 0-1)
        image_array_rgb = image_array_rgb / 255.0

        input_tensor = np.expand_dims(image_array_rgb, axis=0) # Model için batch boyutu ekle: (1, 224, 224, 3)

        predictions = model_usg.predict(input_tensor)
        predicted_class_id = np.argmax(predictions, axis=1)[0]
        prediction_probabilities = predictions[0].tolist()

        # USG modelinden aldığımız çıktı sınıf isimleri
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
    lab_data: str = Form(..., description="JSON formatında Lab verileri (Yaş, Cinsiyet, Albumin, vb.)"),
    usg_file: Union[UploadFile, str] = Form("", description="İsteğe bağlı USG Görüntüsü Dosyası (JPG/PNG) veya boş string"),
    mri_file: Union[UploadFile, str] = Form("", description="İsteğe bağlı MRI Görüntüsü Dosyası (NIfTI veya diğer) veya boş string"),
    afp_value: Optional[float] = Form(None, description="AFP değeri (ng/mL)"), # AFP'yi formdan alacağız
    alcohol_consumption: Optional[str] = Form(None, description="Alkol Tüketimi (Evet/Hayır)"),
    smoking_status: Optional[str] = Form(None, description="Sigara Kullanımı (Evet/Hayır)"),
    hcv_status: Optional[str] = Form(None, description="HCV (Hepatit C) Durumu (Evet/Hayır)"), # Yeni eklendi
    hbv_status: Optional[str] = Form(None, description="HBV (Hepatit B) Durumu (Evet/Hayır)"), # Yeni eklendi
    cancer_history_status: Optional[str] = Form(None, description="Ailede Kanser Öyküsü Durumu (Var/Yok)") # Yeni eklendi
):
    """
    Hasta laboratuvar verileri ve isteğe bağlı görüntüleme verilerini (USG/MRI) alarak
    birleşik HCC riski değerlendirmesi ve takip/tedavi önerisi sunar.
    Bu, projenizin Agentic AI çekirdeğidir.
    """
    overall_risk_level = "Düşük Risk (Karaciğer Hastalığı Belirtisi Yok)" # Varsayılan olarak Düşük Risk
    detailed_report_summary = []
    mri_recommendation = False
    final_recommendation = "HCC riski düşük. Rutin yıllık kontroller (Lab testleri ve USG) önerilir." # Varsayılan düşük risk önerisi
    
    # AFP için varsayılan değer
    afp_val_for_risk_adj = afp_value if afp_value is not None else 0.0

    # Dosya varlığı kontrolleri
    is_usg_file_present = isinstance(usg_file, UploadFile)
    is_mri_file_present = isinstance(mri_file, UploadFile)

    print(f"\n--- Yeni İstek: /evaluate_hcc_risk ---")
    print(f"Alınan Lab Verisi (string): {lab_data}")
    print(f"USG Dosyası Var mı?: {is_usg_file_present}")
    print(f"MRI Dosyası Var mı?: {is_mri_file_present}")
    print(f"AFP Değeri: {afp_val_for_risk_adj}")
    print(f"Alkol Tüketimi: {alcohol_consumption}")
    print(f"Sigara Kullanımı: {smoking_status}")
    print(f"HCV Durumu: {hcv_status}")
    print(f"HBV Durumu: {hbv_status}")
    print(f"Ailede Kanser Öyküsü: {cancer_history_status}")


    # 1. Lab Modelini Çalıştır
    lab_prediction_result = None
    try:
        lab_data_parsed = LabData.model_validate_json(lab_data)
        lab_response = await predict_lab_risk(lab_data_parsed)
        lab_prediction_result = {
            "predicted_disease_label": lab_response["predicted_disease_label"],
            "hcc_probability": lab_response["hcc_probability"],
            "risk_level": lab_response["risk_level"], # Düşük, Orta, Yüksek
            "all_class_probabilities": lab_response["all_class_probabilities"]
        }
        detailed_report_summary.append(f"Laboratuvar Analizi: Tahmin Edilen Hastalık: {lab_prediction_result['predicted_disease_label']}, HCC Riski Seviyesi: {lab_prediction_result['risk_level']} (HCC Olasılığı: %{round(lab_prediction_result['hcc_probability'] * 100, 2)})")

        # Lab modelinin kendi HCC olasılığına göre genel risk seviyesini ata
        overall_risk_level = lab_prediction_result['risk_level']
        # Eğer lab model yüksek risk verdiyse, MRI önermeyi düşün (AFP'den bağımsız)
        if overall_risk_level == "Yüksek Risk (Belirgin Karaciğer Hastalığı Şüphesi)":
            mri_recommendation = True
            detailed_report_summary.append("Lab verileri, belirgin karaciğer hastalığı şüphesi nedeniyle ileri görüntüleme (MRI) gerektirebilir.")

    except Exception as e:
        detailed_report_summary.append(f"HATA: Laboratuvar verileri işlenirken bir sorun oluştu: {e}")
        print(f"HATA: Lab verileri işlenirken: {e}")
        lab_prediction_result = {"predicted_disease_label": "Hata", "hcc_probability": 0, "risk_level": "Hata", "all_class_probabilities":{}}


    # AFP değeri ile riski manuel olarak ayarla (manual_input_predict'teki mantık)
    # Sadece lab modeli sağlıklı veya düşük risk dese bile AFP yüksekse riski yükselt.
    adjusted_hcc_probability = lab_prediction_result["hcc_probability"]
    if afp_val_for_risk_adj > 400:
        adjusted_hcc_probability += 0.10
        detailed_report_summary.append(f"AFP değeri ({afp_val_for_risk_adj}) çok yüksek (400 üzeri) olduğu için HCC olasılığı manuel olarak %10 artırıldı. MRI önerisi tetiklendi.")
        mri_recommendation = True # AFP 400 üzeri ise MRI mutlaka önerilir
    elif afp_val_for_risk_adj > 200:
        adjusted_hcc_probability += 0.05
        detailed_report_summary.append(f"AFP değeri ({afp_val_for_risk_adj}) yüksek (200 üzeri) olduğu için HCC olasılığı manuel olarak %5 artırıldı. MRI önerisi tetiklendi.")
        mri_recommendation = True # AFP 200 üzeri ise MRI mutlaka önerilir
    
    adjusted_hcc_probability = min(adjusted_hcc_probability, 1.0) # Olasılık 1.0'ı geçmesin

    # Ayarlanmış olasılığa göre overall_risk_level'ı güncelle (en son ve baskın karar)
    if adjusted_hcc_probability >= 0.66:
        overall_risk_level = "Yüksek Risk (Belirgin Karaciğer Hastalığı Şüphesi)"
    elif adjusted_hcc_probability >= 0.33:
        overall_risk_level = "Orta Risk (Hafif Karaciğer Hastalığı Şüphesi)"
    else:
        overall_risk_level = "Düşük Risk (Karaciğer Hastalığı Belirtisi Yok)"


    # 2. USG Görüntüsü Varsa USG Modelini Çalıştır
    usg_fibrosis_result = None
    if is_usg_file_present:
        try:
            usg_prediction_response = await predict_usg_fibrosis(usg_file)
            usg_fibrosis_result = {
                "stage_label": usg_prediction_response["predicted_fibrosis_stage_label"],
                "stage_id": usg_prediction_response["predicted_fibrosis_stage_id"]
            }
            detailed_report_summary.append(f"USG Görüntü Analizi: Karaciğer Fibrozis Evresi: {usg_fibrosis_result['stage_label']}")

            # USG fibrozis evresine göre risk değerlendirmesi
            # Dokümanınıza göre F3 veya F4 yüksek riskliydi.
            if usg_fibrosis_result["stage_id"] >= 3: # F3 (Ağır Fibroz) veya F4 (Siroz) ise
                if "Yüksek Risk" not in overall_risk_level: # Eğer zaten yüksek risk değilse yükselt
                    overall_risk_level = "Yüksek Risk (Belirgin Karaciğer Hastalığı Şüphesi)"
                mri_recommendation = True # USG'den ileri fibrozis gelirse MRI önerisi güçlü
                detailed_report_summary.append(f"USG bulguları ({usg_fibrosis_result['stage_label']}), HCC riski için belirgin bir faktör olup ileri görüntüleme (MRI) gerektirebilir.")
            elif usg_fibrosis_result["stage_id"] >= 1: # F1 veya F2 ise
                if "Düşük Risk" in overall_risk_level: # Eğer Lab'dan düşük geldiyse ortaya çek
                    overall_risk_level = "Orta Risk (Hafif Karaciğer Hastalığı Şüphesi)"
                detailed_report_summary.append(f"USG bulguları ({usg_fibrosis_result['stage_label']}), karaciğerde fibrozis belirtileri göstermektedir. Yakın takip önerilir.")
            else: # F0 ise
                 detailed_report_summary.append("USG bulguları (F0-Fibroz yok), karaciğerde fibrozis belirtisi göstermemektedir.")

        except Exception as e:
            detailed_report_summary.append(f"HATA: USG görüntüsü işlenirken bir sorun oluştu: {e}")
            print(f"HATA: USG görüntüsü işlenirken: {e}")
            usg_fibrosis_result = None


    # 3. MRI Görüntüsü Varsa MRI Modelini Çalıştır (Placeholder - Henüz Model Yok)
    mri_tumor_result = None
    if is_mri_file_present:
        detailed_report_summary.append("MRI Görüntüsü yüklendi.")
        # MRI yüklendiği için artık MRI önerisi yapmıyoruz, çünkü zaten çekilmiş demektir.
        mri_recommendation = False
        overall_risk_level = "Yüksek Risk (Belirgin Karaciğer Hastalığı Şüphesi)"
        detailed_report_summary.append("MRI analizi (tümör boyutu, HCC evresi) burada yapılacaktır (MRI modeli entegre edildikten sonra).")

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
    # Koşulları en yüksek risk/ihtiyaçtan en düşüğe doğru sıralayalım.
    if is_mri_file_present: # Eğer MRI dosyası yüklendiyse (yani MRI çekilmişse)
        if mri_tumor_result: # Ve MRI modeli de çalışıp tümör bulduysa (ileride)
            final_recommendation = "HCC tanısı ve evrelemesi yapıldı. Uzman onkolog ile tedavi planlaması önerilir."
        else: # MRI yüklendi ama henüz modeli yok veya tümör analizi sonucu gelmedi (şimdiki durum)
            final_recommendation = "MRI görüntülemesi yapıldı. Detaylı analiz (tümör boyutu, HCC evresi) bekleniyor. Uzman değerlendirmesi önemlidir."
    elif mri_recommendation: # Eğer MRI önerisi Lab veya USG'den geldiyse (MRI henüz çekilmediyse)
        final_recommendation = "HCC riski yüksek. Kesin tanı ve evreleme için MRI görüntülemesi ŞİDDETLE ÖNERİLİR."
    elif "Yüksek Risk" in overall_risk_level: # Sadece Lab veya USG'den yüksek risk gelmiş ve MRI önerisi yoksa (nadiren)
        final_recommendation = "Yüksek düzeyde HCC riski. Uzman gastroenterolog/hepatolog değerlendirmesi ve yakın takip (3 ayda bir AFP ve USG) önerilir."
    elif "Orta Risk" in overall_risk_level:
        final_recommendation = "Orta düzeyde HCC riski. 6 ayda bir AFP ve USG ile yakın takip önerilir. Uzman gastroenterolog/hepatolog değerlendirmesi düşünülebilir."
    else: # overall_risk_level == "Düşük Risk" ise
        final_recommendation = "HCC riski düşük. Rutin yıllık kontroller (Lab testleri ve USG) önerilir."

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