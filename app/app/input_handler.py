import streamlit as st

def get_user_inputs():
    with st.form("formulario_prediccion"):
        st.subheader("Introduce los detalles de la propiedad")

        address = st.text_input("Dirección")
        agency_name = st.text_input("Agencia")
        bathroom_count = st.number_input("Baños", min_value=0)
        bedroom_count = st.number_input("Dormitorios", min_value=0)
        energy_certificate = st.selectbox("Certificado energético", ["A", "B", "C", "D", "E", "F", "G"])
        floor = st.number_input("Piso", step=1)
        latitude = st.number_input("Latitud", format="%.6f")
        longitude = st.number_input("Longitud", format="%.6f")
        lot_size = st.number_input("Tamaño del terreno (m2)", min_value=0)
        property_description = st.text_area("Descripción de la propiedad")
        property_id = st.text_input("ID de la propiedad")
        property_title = st.text_input("Título del anuncio")
        property_type = st.selectbox("Tipo de propiedad", ["flat", "Chalet"])
        neighborhood = st.selectbox("Barrio", ["chamberi", "centro", "arganzuela", "retiro"])
        exterior = st.checkbox("¿Es exterior?", value=False)
        ascensor = st.checkbox("¿Tiene ascensor?", value=False)
        
        uploaded_images = st.file_uploader(
            "Imágenes de la propiedad (puedes subir varias)",
            type=["webp"],
            accept_multiple_files=True
        )

        submit = st.form_submit_button("Predecir")

    if submit:
        return {
            "address": address,
            "agency_name": agency_name,
            "bathroom_count": bathroom_count,
            "bedroom_count": bedroom_count,
            "energy_certificate": energy_certificate,
            "floor": floor,
            "latitude": latitude,
            "longitude": longitude,
            "lot_size": lot_size,
            "property_description": property_description,
            "property_id": property_id,
            "property_title": property_title,
            "property_type": property_type,
            "neighborhood": neighborhood,
            "images": uploaded_images,
            "exterior": int(exterior),
            "ascensor": int(ascensor),
        }

    return None
