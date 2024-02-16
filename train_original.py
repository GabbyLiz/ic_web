from os import listdir
from nltk.translate.bleu_score import corpus_bleu
from numpy import array, argmax
from pickle import load, dump
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import load_img, img_to_array
from keras.utils import to_categorical, plot_model
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
from keras.applications.vgg16 import VGG16, preprocess_input
from tqdm import tqdm
import string


# Extraer características de cada foto del directorio
def extract_features(directory):
    # Cargar el modelo
    model = VGG16()
    # Modificar el modelo
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # model = Model(inputs=model.inputs, outputs=model.layers[-1].output)
    # Resumen
    print(model.summary())
    # Extraer las características de cada una de las fotos
    features = dict()
    for name in listdir(directory):
        # Carga una imagen desde un archivo
        filename = directory + '/' + name
        image = load_img(filename, target_size=(224, 224))
        # Convierte los píxeles de la imagen en una matriz numpy
        image = img_to_array(image)
        # Modifica los datos del modelo
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Prepara la imagen para el modelo VGG
        image = preprocess_input(image)
        # Obtienes las características
        feature = model.predict(image, verbose=0)
        # Obtiene el identificador de imagen
        image_id = name.split('.')[0]
        # Almacena las características
        features[image_id] = feature
        print('>%s' % name)

    return features


## Carga de archivos que contiene todas las descripciones
## Se hace una limpieza del texto cargado
def load_description(filename):
    # Abrir archivo como lectura
    file = open(filename, 'r')
    # Lectura de todo el texto
    doc = file.read()
    # Cerrar el archivo
    file.close()
    """
    Captions dict es de la forma:
    {
        image_id1 : [caption1, caption2, etc],
        image_id2 : [caption1, caption2, etc],
        ...
    }
    """
    # Extraer descripciones de las imagenes
    mapping = dict()
    # Procesar las lineas
    count = 0
    for line in doc.split('\n'):
        # Dividir línea por espacio en blanco
        tokens = line.split()
        if len(line) < 2:
            continue
        # Toma el primer token como el id de la imagen, el resto como la descripción
        image_id, image_caption = tokens[0], tokens[1:]
        # Eliminar el nombre del archivo del id de la imagen
        image_id = image_id.split('.')[0]
        # Volver a convertir los tokens de descripción en cadena
        image_caption = ' '.join(image_caption)
        # Crear la lista si es necesario
        if image_id not in mapping:
            mapping[image_id] = list()
        # Pie de foto
        mapping[image_id].append(image_caption)
        count = count + 1
    print('Subtítulos procesados: ', count)
    return mapping


"""
	Limpieza del texto de las descripciones
	Dado el diccionario de identificadores de imagen a las descripciones
	recorre cada descripcion y limpia el texto
"""


def clean_descriptions(descriptions):
    # Preparar la tabla de traducción para eliminar los signos de puntuación
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
        for i in range(len(desc_list)):
            desc = desc_list[i]
            # Tokenizar, es decir, dividir en espacios en blanco
            desc = desc.split()
            # Convertir en minúsculas
            desc = [word.lower() for word in desc]
            # Eliminar la puntuación de cada token
            desc = [w.translate(table) for w in desc]
            # Quitar 's' y 'a' sobrantes
            desc = [word for word in desc if len(word) > 1]
            # Eliminar tokens con números
            desc = [word for word in desc if word.isalpha()]
            # Guardar como cadena
            desc_list[i] = ' '.join(desc)


# Convierte las descripciones cargadas en un vocabulario de palabras
def to_vocabulary(descriptions):
    # Crear una lista de todas las cadenas de descripción
    all_desc = set()
    for key in descriptions.keys():
        [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc


# Guardar una descripción por linea en un archivo
"""
	- Guardar subtítulos en un archivo, uno por línea
	- Después de guardar, captions.txt tiene la forma : - 'id' 'caption'.
	  Ejemplo : 2252123185_487f21e336 stadium full of people watch game
"""


def save_descriptions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()


# Cargar archivo en memoria
def load_set(filename):
    # Abrir el archivo como solo lectura
    file = open(filename, 'r')
    # Leer todo el texto
    doc = file.read()
    # Cerrar el archivo
    file.close()

    # Cargar una lista predefinida de identificadores de fotos
    dataset = list()
    # Procesar línea por línea
    for line in doc.split('\n'):
        # Omitir líneas vacías
        if len(line) < 1:
            continue
        # Obtener el identificador de la imagen (id)
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


# load clean descriptions into memory
def load_clean_descriptions(filename, dataset):
    # Cargar documento
    file = open(filename, 'r')
    doc = file.read()
    file.close()
    descriptions = dict()
    # Procesar línea por línea
    for line in doc.split('\n'):
        # Dividir línea por espacio en blanco
        tokens = line.split()
        # Separar id de la descripción
        image_id, image_desc = tokens[0], tokens[1:]
        # Omitir imágenes no incluidas en el id en el conjunto
        if image_id in dataset:
            # Crear lista
            if image_id not in descriptions:
                descriptions[image_id] = list()
            # Delimitar subtítulos en fichas de inicio y finalización
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # Almacenamiento
            descriptions[image_id].append(desc)
    return descriptions


# Cargar características de la imagen para un conjunto determinado
def load_photo_features(filename, dataset):
    # Cargar todas las características
    all_features = load(open(filename, 'rb'))
    # Características del filtro
    features = {k: all_features[k] for k in dataset}
    return features


# Convertir el diccionario de descripciones en una lista
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc


# Ajustar un token dadas las descripciones
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


# Calcular la longitud de los pies de foto con más palabras
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(d.split()) for d in lines)


# create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, descriptions, photos, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each image identifier
    for key, desc_list in descriptions.items():
        # walk through each description for the image
        for desc in desc_list:
            # encode the sequence
            seq = tokenizer.texts_to_sequences([desc])[0]
            # split one sequence into multiple X,y pairs
            for i in range(1, len(seq)):
                # split into input and output pair
                in_seq, out_seq = seq[:i], seq[i]
                # pad input sequence
                in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                # encode output sequence
                out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                # store
                X1.append(photos[key][0])
                X2.append(in_seq)
                y.append(out_seq)
    return array(X1), array(X2), array(y)


# Definir el modelo RNN
# La siguiente función denominada define_RNNModel() define y devuelve el
# modelo listo para ajustarse.
def define_model(vocab_size, max_length):
    # modelo extractor de características
    # VGG16 produce un vector de 4096 dimensiones para cada imagen, que alimentaremos al modelo RNN
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Modelo procesador de secuencia
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # Modelo decodificador
    # Fusión de los modelos y creación de un clasificador softmax
    decoder1 = add([fe2, se3])
    decoder2 = Dense(256, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Juntar [imagen, secuencia] [palabra]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # Resumen del modelo
    print(model.summary())
    #plot_model(model, to_file='model.png', show_shapes=True)
    return model


# Asignar un número entero a una palabra
def int_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


# Generar la descripción de una imagen
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
        word = int_to_word(yhat, tokenizer)
        # Parar si no podemos mapear la palabra
        if word is None:
            break
            # Añadir como entrada para generar la siguiente palabra
        in_text += ' ' + word
        # Parar si predecimos el final de la secuencia
        if word == 'endseq':
            break
    return in_text


"""
	Evaluar el modelo en la puntuación BLEU utilizando predicciones argmax.
"""

# Validar con los datos de prueba
def evaluate_model(model, descriptions, photos, tokenizer, max_length):
    actual, predicted = list(), list()
    for key, desc_list in tqdm(descriptions.items()):
        # Predecir el título de la imagen
        yhat = generate_desc(model, tokenizer, photos[key], max_length)
        # Dividir en palabras
        ground_truth = [caption.split() for caption in desc_list]
        # Agregar a la lista
        actual.append(ground_truth)
        predicted.append(yhat.split())
    # Calcular la puntuación BLEU
    print('Puntaje BLEU :')
    print(
        'Una coincidencia perfecta da como resultado una puntuación de 1,0, mientras que una falta de coincidencia perfecta da como resultado una puntuación de 0,0.')
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# LLAMADA DE METODOS
# Primera parte: extraer características de todas las imágenes
directoryDataset = 'train_val_data/Flickr8k_Dataset/'
features = extract_features(directoryDataset)
print('Características extraídas: %d' % len(features))
# Guardar en un archivo
dump(features, open('features.pkl', 'wb'))

# Segunda parte
directoryToken = 'train_val_data/Flickr8k.token.txt'
# Cargar las descripciones
descriptions = load_description(directoryToken)
print('Cargados: %d ' % len(descriptions))
# Limpiar las descripciones de los textos
clean_descriptions(descriptions)
# Resumir el vocabulario
vocabulary = to_vocabulary(descriptions)
print('Tamaño del vocabulario: %d' % len(vocabulary))
# Guardar en un archivo
save_descriptions(descriptions, 'descriptions.txt')

# Tercera parte: Dataset de entrenamiento

# Cargar el conjunto de datos de entrenamiento(6K)
filename = 'train_val_data/Flickr_8k.trainImages.txt'
train = load_set(filename)
print('Dataset: %d' % len(train))
# Descripciones
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descripcioness: entrenamiento=%d' % len(train_descriptions))
# Características de las fotos
train_features = load_photo_features('features.pkl', train)
print('Fotos: entrenamiento=%d' % len(train_features))
# Preparar tokenizador
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print('Tamaño del vocabulario: %d' % vocab_size)
# Determinar la longitud máxima de la secuencia
max_length = max_length(train_descriptions)
print('Longitud de descripción: %d' % max_length)
# Preparar secuencias de entrenamiento
X1train, X2train, ytrain = create_sequences(tokenizer, max_length, train_descriptions, train_features, vocab_size)

# Cuarta Parte: dev dataset

# Cargar el conjunto de datos de prueba
filename = 'train_val_data/Flickr_8k.devImages.txt'
test = load_set(filename)
print('Dataset: %d' % len(test))
# Descripciones
test_descriptions = load_clean_descriptions('descriptions.txt', test)
print('Descripciones: prueba=%d' % len(test_descriptions))
# Características de las fotos
test_features = load_photo_features('features.pkl', test)
print('Fotos: prueba=%d' % len(test_features))
# Preparar las secuencias de prueba
X1test, X2test, ytest = create_sequences(tokenizer, max_length, test_descriptions, test_features, vocab_size)

# Quinta parte: Entrenar el modelo

# Definir el modelo
model = define_model(vocab_size, max_length)
# Definir callback de punto de control
filepath = 'model-ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5'
checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# Modelo de ajuste
model.fit([X1train, X2train], ytrain, epochs=5, verbose=2, callbacks=[checkpoint],
          validation_data=([X1test, X2test], ytest))

# Evalar el modelo con los datos de validación y obtener la puntuación BLEU
evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)
