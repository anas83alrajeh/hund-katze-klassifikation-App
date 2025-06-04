import streamlit as st
from PIL import Image
import io
import requests

# Listen mit Hunderassen und Katzenarten (alles klein geschrieben)
hunde = [hund.lower() for hund in [
    'chihuahua', 'golden_retriever', 'labrador_retriever', 'french_bulldog', 
    'beagle', 'pomeranian', 'rottweiler', 'yorkshire_terrier', 'boxer', 'dalmatian',
    'great_dane', 'shih-tzu', 'bull_mastiff', 'papillon', 'siberian_husky',
    'pug', 'border_collie', 'basset', 'doberman', 'affenpinscher',
]]

katzen = [katze.lower() for katze in [
    'tabby', 'tiger_cat', 'persian_cat', 'siamese_cat', 'egyptian_cat',
    'lynx', 'leopard', 'snow_leopard', 'jaguar', 'lion',
]]

st.title("Bildklassifikation: Hund, Katze oder etwas anderes")
st.write("Lade ein Bild hoch, um zu erkennen, ob es ein Hund, eine Katze oder etwas anderes ist.")

hochgeladene_datei = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if hochgeladene_datei is not None:
    bild = Image.open(hochgeladene_datei).convert('RGB')
    st.image(bild, caption='Hochgeladenes Bild', use_column_width=True)

    buffered = io.BytesIO()
    bild.save(buffered, format="JPEG")
    img_bytes = buffered.getvalue()

    api_url = "https://api-inference.huggingface.co/models/google/vit-base-patch16-224"
    headers = {"Authorization": "Bearer DEIN_HUGGINGFACE_API_TOKEN"}

    with st.spinner("Modell wird geladen und Bild wird klassifiziert..."):
        response = requests.post(api_url, headers=headers, data=img_bytes)

    if response.status_code == 200:
        ergebnisse = response.json()
        label = ergebnisse[0]['label']
        score = ergebnisse[0]['score'] * 100
        label_lower = label.lower()

        if label_lower in hunde:
            st.success(f"Das ist ein Hund ({label}), Vertrauen: {score:.2f}%")
        elif label_lower in katzen:
            st.success(f"Das ist eine Katze ({label}), Vertrauen: {score:.2f}%")
        else:
            st.error(f"Das ist weder eine Katze noch ein Hund (erkannt als: {label}, Vertrauen: {score:.2f}%)")
    else:
        st.error("Fehler beim Modell-API-Aufruf. Bitte API-Token und Internetverbindung prüfen.")

st.markdown("---")
st.markdown("Erstellt von: Anas Al Rajeh  \n Kontakt: [anasalrajeh9@gmail.com](mailto:anasalrajeh9@gmail.com)")
