# libreria para traduccion de texto
from googletrans import Translator
# libreria para transformación de audio
from gtts import gTTS
from io import BytesIO
from pickle import load
from PIL import Image
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.sequence import pad_sequences
import numpy as np
translator = Translator()
# Libreria para colcoar imagen de fondo
import base64
# Libreria para desarrollar aplicaciones con Streamlit
import streamlit as st


# Define the CNN model

def CNNModel(model_type):
    if model_type == 'inceptionv3':
        model = InceptionV3()
    elif model_type == 'vgg16':
        model = VGG16()
    elif model_type == 'resnet50':
        model = ResNet50()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    return model


@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img


# Extraer características de la imagen
@st.cache(allow_output_mutation=True)
def extract_features(filename, model, model_type):
    if model_type == 'inceptionv3':
        from keras.applications.inception_v3 import preprocess_input
        target_size = (299, 299)
    elif model_type == 'resnet50':
        from keras.applications.resnet50 import preprocess_input
        target_size = (224, 224)
    elif model_type == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
        target_size = (224, 224)
    # Cargar y redimensionar la imagen
    image = load_image(filename)
    image = image.resize(target_size)

    # Convertir la imagen de pixeles a numpy array
    image = img_to_array(image)
    # Redimensionar datos para el modelo
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    # Preparar la imagen para el modelo CNN
    image = preprocess_input(image)
    # Pasar la imagen al modelo para obtener características codificadas
    features = model.predict(image, verbose=0)
    return features

def generate_caption_beam_search(model, tokenizer, image, max_len, beam_index=3):
    # start_word --> [[idx,prob]] ;prob=0 initially
    start_word = [[tokenizer.texts_to_sequences(['startseq'])[0], 0.0]]
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_len)
            preds = model.predict([image,par_caps], verbose=0)
            # Tomar las mejores predicciones `beam_index` (es decir, las que tienen mayores probabilidades)
            word_preds = np.argsort(preds[0])[-beam_index:]

            # Crear una nueva lista para volver a pasarlos por el modelo
            for word in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(word)
                # Actualizar probabilidad
                prob += preds[0][word]
                #  Añadir como entrada para generar la siguiente palabra
                temp .append([next_cap, prob])

        start_word = temp
        # Ordenar según las probabilidades
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Tomar las palabras principales
        start_word = start_word[-beam_index:]


    start_word = start_word[-1][0]
    intermediate_caption = [int_to_word(i,tokenizer) for i in start_word]

    final_caption = []

    for word in intermediate_caption:
        if word=='endseq':
            break
        else:
            final_caption.append(word)

    final_caption.append('endseq')
    return ' '.join(final_caption)


# Asigna un número entero a una palabra

def int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


@st.cache
def image_caption(imagen, model_type, model_load_path, tokenizer_path1):
    # # Cargar el tokenizador
    #tokenizer_path = config['tokenizer_path']
    tokenizer_path = tokenizer_path1
    tokenizer = load(open(tokenizer_path, 'rb'))

    # Longitud máxima de la secuencia (de entrenamiento)
    max_length = 51

    # # Carga el modelo
    caption_model = load_model(model_load_path)

    image_model = CNNModel(model_type)
    # Codificación de la imagen mediante el modelo CNN
    image = extract_features(imagen, image_model, model_type)
    # Generación de subtítulos mediante modelo RNN decodificador + búsqueda BEAM
    generated_caption = generate_caption_beam_search(caption_model, tokenizer, image, max_length,beam_index=3)

    # Generación de subtítulos mediante modelo RNN decodificador + Argmax
    #generated_caption = generate_caption(caption_model, tokenizer, image, max_length)
    # Quitar startseq y endseq
    caption = 'Image description: ' + generated_caption.split()[1].capitalize()
    for x in generated_caption.split()[2:len(generated_caption.split()) - 1]:
        caption = caption + ' ' + x
    caption += '.'

    return caption

def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
    hide_streamlit_style = """
                <style>
                footer {visibility: hidden;}
                </style>
                """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)
    # ---Estructura de la página---

    # Título
    st.title("Detección de objetos y semántica de una imagen")
    st.header("_Ingrese una imagen para realizar la descripción:_")

## Configurar la barra lateral
    st.sidebar.title('Modelos')
    st.sidebar.subheader(
        'Los modelos pre-entrenados varían la calidad de detección, dependiendo de la configuración de cada uno en el entrenamiento de la red')

    app_mode = st.sidebar.selectbox('Seleccione uno de los siguientes modelos para la detección',
                                    ['Modelo 1', 'Modelo 2', 'Modelo 3', 'Modelo 4'])
    if app_mode == 'Modelo 1':
        model_type = 'inceptionv3'
        model_load_path = 'model_data/m11_model_inceptionv3_epoch-04_train_loss-2.5424_val_loss-2.8094.hdf5'
        tokenizer_path1 = 'model_data/m11_tokenizer_inception.pkl'

    if app_mode == 'Modelo 2':
        model_type = 'resnet50'
        model_load_path = 'model_data/m12_model_resnet50_epoch-06_train_loss-2.2868_val_loss-2.8316.hdf5'
        tokenizer_path1 = 'model_data/m12_tokenizer_resnet50_coco.pkl'

    if app_mode == 'Modelo 3':
        model_type = 'vgg16'
        model_load_path = 'model_data/m14_model_vgg16_epoch-06_train_loss-2.4527_val_loss-2.9014.hdf5'
        tokenizer_path1 = 'model_data/m14_tokenizer_vgg16_coco.pkl'

    if app_mode == 'Modelo 4':
        model_type = 'vgg16'
        model_load_path = 'model_data/model_vgg16_epoch-08_train_loss-2.2872_val_loss-2.8882.hdf5'
        tokenizer_path1 = 'model_data/tokenizer.pkl'

    # Solicitar al usuario una imagen
    uploaded_image = st.file_uploader("Seleccione una imagen de su dispositivo...", type=["jpg", "jpeg", "webp"])

    if uploaded_image is not None:
        img = load_image(uploaded_image)
        st.image(img, caption='Imagen Cargada', use_column_width=True)

        st.success("¡Imagen Cargada!")
        descripcion = image_caption(uploaded_image, model_type, model_load_path, tokenizer_path1)

        translation = translator.translate(descripcion, dest='es')
        caption = translation.text

        st.write(caption)

        st.info("Generando audio, por favor espere...")

        language = 'es'
        text = str(caption)

        sound_file = BytesIO()
        myobj = gTTS(text=text, lang=language, slow=False)
        myobj.write_to_fp(sound_file)

        st.audio(sound_file)

add_bg_from_local('logo2.png')