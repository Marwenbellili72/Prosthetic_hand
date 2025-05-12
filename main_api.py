import os
import io
import numpy as np
import joblib
import xgboost as xgb
from scipy.io import loadmat
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
import traceback 

MODEL_PATH = "emg_xgboost_model.json"
SCALER_PATH = "emg_scaler.pkl"
LABEL_MAP_PATH = "label_map_inverse.pkl" 
TEMPLATES_DIR = "templates" 
STATIC_DIR = "static"       

WINDOW_SIZE = 200
OVERLAP = 100 
N_FEATURES_PER_CHANNEL = 5 

app = FastAPI(title="API de Prédiction de Mouvements EMG + Visu 3D", version="1.1")

model = None
scaler = None
label_map_inverse = None
load_error_message = None 

print("Chargement du modèle, du scaler et du mapping...")
try:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Fichier modèle non trouvé: {MODEL_PATH}")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    print(f"Modèle chargé depuis: {MODEL_PATH}")

    if not os.path.exists(SCALER_PATH):
         raise FileNotFoundError(f"Fichier scaler non trouvé: {SCALER_PATH}")
    scaler = joblib.load(SCALER_PATH)
    print(f"Scaler chargé depuis: {SCALER_PATH}")

    if LABEL_MAP_PATH and os.path.exists(LABEL_MAP_PATH):
        label_map_inverse = joblib.load(LABEL_MAP_PATH)
        print(f"Mapping d'étiquettes chargé depuis: {LABEL_MAP_PATH}")
    elif LABEL_MAP_PATH:
         print(f"AVERTISSEMENT: Fichier de mapping '{LABEL_MAP_PATH}' spécifié mais non trouvé.")

except FileNotFoundError as e:
    load_error_message = f"ERREUR CRITIQUE: {e}. L'API ne pourra pas effectuer de prédictions."
    print(load_error_message)
    model = None 
    scaler = None
except Exception as e:
    load_error_message = f"ERREUR CRITIQUE lors du chargement des artefacts: {e}"
    print(load_error_message)
    model = None
    scaler = None

templates = None
if not os.path.isdir(TEMPLATES_DIR):
    print(f"AVERTISSEMENT: Le répertoire templates '{TEMPLATES_DIR}' n'existe pas. Le rendu HTML via Jinja2 échouera.")
else:
    templates = Jinja2Templates(directory=TEMPLATES_DIR)

if not os.path.isdir(STATIC_DIR):
     print(f"AVERTISSEMENT: Le répertoire static '{STATIC_DIR}' n'existe pas.")
     print("Le modèle 3D, JS ou CSS pourraient ne pas être servis.")
else:
     app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
     print(f"Répertoire static monté sur /static")


def extract_features(emg_data, window_size=WINDOW_SIZE, overlap=OVERLAP):
    """
    Extrait les descripteurs temporels des signaux EMG.
    """
    if emg_data is None or not isinstance(emg_data, np.ndarray) or emg_data.ndim == 0:
         raise ValueError("Données EMG invalides fournies à extract_features.")

    if emg_data.ndim == 1:
        emg_data = emg_data.reshape(-1, 1)

    n_samples, n_channels = emg_data.shape

    if n_samples < window_size:
        print(f"Avertissement: Données trop courtes ({n_samples}) pour window_size ({window_size}).")
        return np.empty((0, n_channels * N_FEATURES_PER_CHANNEL)) 

    step = window_size - overlap
    if step <= 0:
        raise ValueError("La taille de la fenêtre doit être supérieure au chevauchement.")

    n_windows = (n_samples - window_size) // step + 1
    if n_windows <= 0:
        return np.empty((0, n_channels * N_FEATURES_PER_CHANNEL))

    features = np.zeros((n_windows, n_channels * N_FEATURES_PER_CHANNEL))

    for w in range(n_windows):
        start = w * step
        end = start + window_size
        window = emg_data[start:end, :]
        feature_col_idx = 0
        for c in range(n_channels):
            channel_data = window[:, c]
            if np.all(channel_data == channel_data[0]):
                mav, rms, wl, zc, ssc = np.abs(channel_data[0]), np.abs(channel_data[0]), 0.0, 0.0, 0.0
            else:
                mav = np.mean(np.abs(channel_data))
                rms = np.sqrt(np.mean(np.square(channel_data)))
                wl = np.sum(np.abs(np.diff(channel_data)))
                threshold = 1e-6 
                zc = np.sum(((channel_data[:-1] * channel_data[1:]) < 0) & (np.abs(channel_data[:-1] - channel_data[1:]) > threshold))
                diff_signal = np.diff(channel_data)
                ssc = np.sum(((diff_signal[:-1] * diff_signal[1:]) < 0) & (np.abs(diff_signal[:-1]) > threshold) & (np.abs(diff_signal[1:]) > threshold))

            features[w, feature_col_idx : feature_col_idx + N_FEATURES_PER_CHANNEL] = [mav, rms, wl, zc, ssc]
            feature_col_idx += N_FEATURES_PER_CHANNEL

    return np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    """Sert la page HTML principale (index.html)."""
    if templates:
        return templates.TemplateResponse("index.html", {"request": request, "load_error_message": load_error_message})
    else:
        html_fallback_path = "index.html"
        if os.path.exists(html_fallback_path):
             with open(html_fallback_path, "r", encoding="utf-8") as f:
                 html_content = f.read()
             if load_error_message:
                 error_div = f'<div class="error"><strong>Erreur Configuration:</strong> {load_error_message}</div>'
                 html_content = html_content.replace("<body>", f"<body>{error_div}")
             return HTMLResponse(content=html_content)
        else:
             return HTMLResponse("<html><body><h1>Erreur</h1><p>Template ou fichier index.html introuvable.</p></body></html>", status_code=500)

@app.post("/predict/", response_class=JSONResponse)
async def predict_movement(file: UploadFile = File(...)):
    """Reçoit un fichier .mat, prédit les mouvements et renvoie les résultats."""
    global model, scaler, label_map_inverse 

    if model is None or scaler is None:
        raise HTTPException(status_code=503, 
                            detail=f"Service indisponible: Modèle ou Scaler non chargé. Raison: {load_error_message or 'Inconnue'}")

    if not file.filename.endswith('.mat'):
        raise HTTPException(status_code=400, detail="Format de fichier invalide. Veuillez uploader un fichier .mat.")

    print(f"Traitement du fichier: {file.filename}")

    try:
        contents = await file.read()
        if not contents:
             raise ValueError("Le fichier reçu est vide.")

        mat_data = loadmat(io.BytesIO(contents))

        emg_key = 'emg'
        if emg_key not in mat_data:
            keys_found = list(mat_data.keys())
            raise KeyError(f"Clé '{emg_key}' non trouvée dans le fichier .mat. Clés disponibles: {keys_found}")

        emg_signal = mat_data[emg_key]
        print(f"Signal EMG extrait, shape: {emg_signal.shape if isinstance(emg_signal, np.ndarray) else 'Invalide'}")

        if not isinstance(emg_signal, np.ndarray) or emg_signal.ndim < 1 or emg_signal.shape[0] < 1:
             raise ValueError("Les données EMG extraites ('emg') sont invalides ou vides.")
        if emg_signal.shape[0] < WINDOW_SIZE:
             raise ValueError(f"Signal EMG trop court ({emg_signal.shape[0]} échantillons) pour la taille de fenêtre ({WINDOW_SIZE}).")

        features = extract_features(emg_signal, window_size=WINDOW_SIZE, overlap=OVERLAP)
        if features.shape[0] == 0:
            raise ValueError("Aucune feature n'a pu être extraite du signal (peut-être trop court après fenêtrage?).")
        print(f"Features extraites, shape: {features.shape}")

        features_scaled = scaler.transform(features)
        print(f"Features mises à l'échelle, shape: {features_scaled.shape}")

        predictions_mapped = model.predict(features_scaled) 
        print(f"Prédictions brutes du modèle: {predictions_mapped}")

        if label_map_inverse:
            try:
                predictions_final = [label_map_inverse[p] for p in predictions_mapped]
            except KeyError as e:
                print(f"ERREUR de Mapping Inverse: Prédiction {e} inconnue dans le mapping. Retour des prédictions brutes.")
                predictions_final = predictions_mapped.tolist()
        else:
            predictions_final = predictions_mapped.tolist()

        print(f"Prédictions finales (après mapping si applicable): {predictions_final}")

        majority_movement = None
        if predictions_final:
            counts = {}
            for p in predictions_final:
                counts[p] = counts.get(p, 0) + 1
            if counts:
                majority_movement = max(counts, key=counts.get)

        response_data = {
            "filename": file.filename,
            "num_windows_processed": features.shape[0],
            "predicted_movements_sequence": predictions_final, 
            "majority_predicted_movement": majority_movement   
        }
        return JSONResponse(content=response_data)

    except (KeyError, ValueError) as e: 
        print(f"Erreur de données/format: {e}")
        raise HTTPException(status_code=400, detail=f"Erreur dans les données fournies ou le format du fichier .mat: {e}")
    except FileNotFoundError as e: 
         print(f"Erreur FNF inattendue: {e}")
         raise HTTPException(status_code=500, detail=f"Erreur serveur: Fichier interne manquant ({e})")
    except Exception as e:
        print(f"ERREUR INATTENDUE lors de la prédiction pour {file.filename}:")
        traceback.print_exc() 
        raise HTTPException(status_code=500, detail=f"Erreur interne du serveur lors du traitement du fichier. Détails dans les logs.")
    finally:
        await file.close()


if __name__ == "__main__":
    import uvicorn
    print("--- Lancement de l'API EMG Prediction + Visu 3D ---")
    print(f"Modèle chargé: {'Oui' if model else 'Non'}")
    print(f"Scaler chargé: {'Oui' if scaler else 'Non'}")
    print(f"Mapping chargé: {'Oui' if label_map_inverse else 'Non'}")
    if load_error_message:
        print(f"ERREUR AU CHARGEMENT: {load_error_message}")
    print(f"Accédez à l'interface via: http://127.0.0.1:8000")
    print(f"Les fichiers statiques (modèle 3D) sont servis depuis le répertoire '{STATIC_DIR}' via '/static/'")

    uvicorn.run("main_api:app", host="127.0.0.1", port=8000, reload=True)
