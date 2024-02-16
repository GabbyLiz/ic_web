import os
import numpy as np
from numpy import argmax
from PIL import Image
from pickle import load
import matplotlib.pyplot as plt
# Keras
from keras.models import Model
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.sequence import pad_sequences
# libreria para traduccion de texto
from googletrans import Translator

translator = Translator()

# Inicializar los valores para entrenamiento de la red
testConfig = {
    'max_length': 40,  # Se establece manualmente tras el entrenamiento del modelo, en el caso de flick8 el valor es 51,34 o 40 y en el caso de coco es 40
    'beam_search_k': 3,
    'tokenizer_path': 'tokenizer_flick.pkl',
    'model_load_path': 'model_inceptionv3_epoch-07_train_loss-2.5311_val_loss-3.1619.hdf5',  # Colocar la ruta del modelo generado con el entrenamiento
    'test_data_path': 'test_data_flickr8/',
    'model_type': 'inceptionv3'  # inceptionv3, vgg16 or resnet50
}

# Definir el modelo CNN (inceptionv3, vgg16, resnet50)
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


# Asignar un número entero a una palabra
def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Generar un pie de foto para una imagen, dado un modelo pre-entrenado y un tokenizador para mapear enteros a palabras.
# Utiliza el algoritmo de búsqueda BEAM
def generate_caption_beam_search(model, tokenizer, image, max_len, beam_index=3):
    start_word = [[tokenizer.texts_to_sequences(['startseq'])[0], 0.0]]
    while len(start_word[0][0]) < max_len:
        temp = []
        for s in start_word:
            par_caps = pad_sequences([s[0]], maxlen=max_len)
            preds = model.predict([image, par_caps], verbose=0)
            # Tomar las mejores predicciones `beam_index` (es decir, las que tienen mayores probabilidades)
            word_preds = np.argsort(preds[0])[-beam_index:]

            # Crear una nueva lista para volver a pasarlos por el modelo
            for word in word_preds:
                next_cap, prob = s[0][:], s[1]
                next_cap.append(word)
                # Actualizar probabilidad
                prob += preds[0][word]
                #  Añadir como entrada para generar la siguiente palabra
                temp.append([next_cap, prob])

        start_word = temp
        # Ordenar según las probabilidades
        start_word = sorted(start_word, reverse=False, key=lambda l: l[1])
        # Tomar las palabras principales
        start_word = start_word[-beam_index:]

    start_word = start_word[-1][0]
    intermediate_caption = [word_for_id(i, tokenizer) for i in start_word]

    final_caption = []

    for word in intermediate_caption:
        if word == 'endseq':
            break
        else:
            final_caption.append(word)

    final_caption.append('endseq')
    return ' '.join(final_caption)


# Generar un pie de foto para una imagen, a partir de un modelo previamente entrenado y un tokenizador para convertir enteros en palabras.
# Utiliza argmax simple
def generate_desc(model, tokenizer, photo, max_length):
    # Iniciar el proceso de generación
    in_text = 'startseq'
    # Recorrer toda la longitud de la secuencia
    for i in range(max_length):
        # Codificar con enteros la secuencia de entrada
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # Rellena las secuencias de entrada
        sequence = pad_sequences([sequence], maxlen=max_length)
        # Predecir la palabra siguiente
        yhat = model.predict([photo, sequence], verbose=0)
        # Convertir la probabilidad en un número entero
        # La función le ayuda a encontrar el índice del máximo en matrices
        yhat = argmax(yhat)
        # Asignar un entero a palabra
        word = word_for_id(yhat, tokenizer)
        # Parar si no podemos mapear la palabra
        if word is None:
            break
            # Añadir como entrada para generar la siguiente palabra
        in_text += ' ' + word
        # Parar si predecimos el final de la secuencia
        if word == 'endseq':
            break
    return in_text


# Extraer características de las imágenes
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
    image = load_img(filename)
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


# Cargar tokenizer
tokenizer = load(open(testConfig['tokenizer_path'], 'rb'))
# Cargar el modelo
caption_model = load_model(testConfig['model_load_path'])
image_model = CNNModel(testConfig['model_type'])

# Cargar y prepara las imágenes
for image_file in os.listdir(testConfig['test_data_path']):
    if image_file.split('--')[0] == 'output':
        continue
    if image_file.split('.')[1] == 'jpg' or image_file.split('.')[1] == 'jpeg':
        print('Generando subtítulo para {}'.format(image_file))
        # Codificación de la imagen con modelo CNN
        image = extract_features(testConfig['test_data_path'] + image_file, image_model, testConfig['model_type'])
        # Genera la descripción usando el decodificador RNN + BEAM search
        generated_caption = generate_caption_beam_search(caption_model, tokenizer, image, testConfig['max_length'], beam_index = testConfig['beam_search_k'])
        # Genera la descripción usando el decodificador RNN + ARGMAX
        #generated_caption = generate_desc(caption_model, tokenizer, image, testConfig['max_length'])
        # Remover startseq y endseq
        caption = 'Caption: ' + generated_caption.split()[1].capitalize()
        for x in generated_caption.split()[2:len(generated_caption.split()) - 1]:
            caption = caption + ' ' + x
        caption += '.'
        # Traducir al español el texto generado
        translation = translator.translate(caption, dest='es')
        caption = translation.text

        # Mostrar la imagen y su descripción
        pil_im = Image.open(testConfig['test_data_path'] + image_file, 'r')
        fig, ax = plt.subplots(figsize=(20, 20))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        _ = ax.imshow(np.asarray(pil_im), interpolation='nearest')
        # _ = ax.set_title("BEAM Search with k={}\n{}".format(config['beam_search_k'],caption),fontdict={'fontsize': '20','fontweight' : '40'})
        _ = ax.set_title(
            "Resultado{}\n{}".format('', caption),
            fontdict={'fontsize': '20', 'fontweight': '40'})
        plt.savefig(testConfig['test_data_path'] + 'salida--' + image_file)
