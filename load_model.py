import joblib
import requests
import os

def get_model(model_path):
    
    try:
        with open(model_path, "rb") as mh:
            rf = joblib.load(mh)
    except:
        print("Cannot fetch model from local downloading from drive")
        if not 'TMLC_severity model.pkl' in os.listdir('.'):
            # example url: "https://drive.google.com/u/1/uc?id=18IxYOI-whucBTZmt5qTvvYgjlxleaSqO&export=download&confirm=t"
            url = "https://drive.google.com/uc?export=download&id=1gDnHwv7MPmqwLLi7MfOVR4bd0axaDv4F"
            r = requests.get(url, allow_redirects=True)
            open(r"TMLC_severity model.pkl", 'wb').write(r.content)
            del r
        with open(r"TMLC_severity model.pkl", "rb") as m:
            rf = joblib.load(m)
    return rf

