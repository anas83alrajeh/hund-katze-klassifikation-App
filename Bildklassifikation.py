import streamlit as st
from PIL import Image
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Definition der Listen mit Hunderassen und Katzenarten (alles klein geschrieben)
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

# Laden des vortrainierten MobileNetV2-Modells
modell = MobileNetV2(weights='imagenet')

st.title("Bildklassifikation: Hund, Katze oder etwas anderes")
st.write("Lade ein Bild hoch, um zu erkennen, ob es ein Hund, eine Katze oder etwas anderes ist.")

# Datei-Upload-Feld für Bilder
hochgeladene_datei = st.file_uploader("Bild hochladen", type=["jpg", "jpeg", "png"])

if hochgeladene_datei is not None:
    # Bild öffnen und in RGB konvertieren
    bild = Image.open(hochgeladene_datei).convert('RGB')
    st.image(bild, caption='Hochgeladenes Bild', use_column_width=True)

    # Bild für das Modell vorbereiten
    bild_resized = bild.resize((224, 224))
    bild_array = img_to_array(bild_resized)
    bild_array = np.expand_dims(bild_array, axis=0)
    bild_array = preprocess_input(bild_array)

    # Vorhersage mit dem Modell
    vorhersage = modell.predict(bild_array)
    entschluesselte_vorhersage = decode_predictions(vorhersage, top=1)[0][0]  # Beste Vorhersage

    vorhersage_label = entschluesselte_vorhersage[1]  # Name der Klasse
    vertrauen = entschluesselte_vorhersage[2] * 100  # Vertrauen in %

    vorhersage_label_klein = vorhersage_label.lower()

    # Einordnung der Vorhersage in Hund, Katze oder anderes
    if vorhersage_label_klein in hunde:
        st.success(f" Das ist ein Hund ({vorhersage_label}), Vertrauen: {vertrauen:.2f}%")
    elif vorhersage_label_klein in katzen:
        st.success(f" Das ist eine Katze ({vorhersage_label}), Vertrauen: {vertrauen:.2f}%")
    else:
        st.error(f" Das ist weder eine Katze noch ein Hund (erkannt als: {vorhersage_label}, Vertrauen: {vertrauen:.2f}%)")
