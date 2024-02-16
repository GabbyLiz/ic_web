from tqdm import tqdm
from numpy import array
from pickle import load, dump
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, concatenate, RepeatVector, TimeDistributed, \
    Bidirectional
from keras.layers.merge import add
from keras.callbacks import ModelCheckpoint
# Para medir la puntuación BLEU
from nltk.translate.bleu_score import corpus_bleu
import random
import os
import string
import numpy as np

# Inicializar los valores para entrenamiento de la red
trainConfig = {
    'embedding_size': 256,
    'LSTM_units': 256,
    'dense_units': 256,
    'dropout': 0.5,
    'num_of_epochs': 1,
    'batch_size': 8,
    'random_seed': 1035,
    'model_type': 'inceptionv3',  # 'vgg16', 'inceptionv3', 'resnet50'
    'directoryDataset': 'train_val_data/Flickr8k_Dataset/',  # 'train_val_data/Flickr8k_Dataset/', 'train_val_data/Coco_Dataset/'
    'directoryToken': 'train_val_data/Flickr8k.token.txt', # 'train_val_data/Flickr8k.token.txt', 'train_val_data/coco.token.txt'
    'directoryTrain': 'train_val_data/Flickr_8k.trainImages.txt',  # 'train_val_data/Flickr_8k.trainImages.txt','train_val_data/coco.trainImages_final.txt'
    'directoryDev': 'train_val_data/Flickr_8k.devImages.txt'
}


def cnn_model(model_type):
    if model_type == 'inceptionv3':
        model = InceptionV3()
    elif model_type == 'vgg16':
        model = VGG16()
    elif model_type == 'resnet50':
        model = ResNet50()
    model.layers.pop()
    model = Model(inputs=model.inputs, outputs=model.layers[-1].output)

    # plot_model(model, to_file="inception_model.png", show_shapes=True)

    return model


# Extraer características de cada foto
def extract_features(directory, model_type):
    if model_type == 'inceptionv3':
        from keras.applications.inception_v3 import preprocess_input
        target_size = (299, 299)
    elif model_type == 'vgg16':
        from keras.applications.vgg16 import preprocess_input
        target_size = (224, 224)
    elif model_type == 'resnet50':
        from keras.applications.resnet50 import preprocess_input
        target_size = (224, 224)
    # Definir modelo CNN, ya sea inceptionv3, vgg16 o resnet50
    model = cnn_model(model_type)
    # Extraer características de cada foto
    features = dict()
    for name in tqdm(os.listdir(directory)):
        # Cargar y redimensionar la imagen dependiendo el modelo de CNN
        filename = directory + name
        image = load_img(filename, target_size=target_size)
        # Convertir los pixeles de la imagen en array numpy
        image = img_to_array(image)
        # Redimensionar la información para el modelo
        image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
        # Preparar la imagen para el modelo CNN seleccionado para obtener características codificadas
        image = preprocess_input(image)
        # Almacenar características codificadas para la imagen
        feature = model.predict(image, verbose=0)
        # Obtener el identificador de la imagen
        image_id = name.split('.')[0]
        features[image_id] = feature
    return features


# Cargar archivos que contienen todas las descripciones
# Se hace una limpieza del texto cargado
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


# Limpieza del texto de las descripciones
# Dado el diccionario de identificadores de imagen a las descripciones
# recorre cada descripcion y limpia el texto


def clean_descriptions(descriptions):
    #  Preparar la tabla de traducción para eliminar los signos de puntuación
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


'''
	Para cargar los datos para los conjuntos de datos de entrenamiento y desarrollo y
	 transformar los datos cargados en pares de entrada-salida para adaptarse a un modelo 
	 de aprendizaje profundo.

	- Disponemos de los archivos Flickr_8k.trainImages.txt y Flickr_8k.devImages.txt que contienen identificadores únicos (id) 
		que se pueden utilizar para filtrar las imágenes y sus descripciones
	- Cargar una lista predefinida de identificadores de imagen(id)
	- Como se ve el expediente:
		2513260012_03d33305cf.jpg
		2903617548_d3e38d7f88.jpg
		3338291921_fe7ae0c8f8.jpg
		488416045_1c6d903fe0.jpg
		2644326817_8f45080b87.jpg
'''


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


'''
	- El modelo que desarrollaremos generará un pie de foto para una imagen dada y el pie de foto se generará palabra a palabra. 
	- La secuencia de palabras generadas previamente se proporcionará como entrada. Por lo tanto, necesitaremos una 'primera palabra' para 
		para iniciar el proceso de generación y una "última palabra" para señalar el final de la leyenda.
	- Se utilizaran las cadenas "startseq" y "endseq". Estos tokens se añaden a los subtítulos a medida que se cargan. 
	- Es importante hacer esto ahora antes de codificar el texto para que los tokens también se codifiquen correctamente.
	- Cargar subtítulos en memoria
	- Vista del archivo:
		1000268201_693b08cb0e child in pink dress is climbing up set of stairs in an entry way
		1000268201_693b08cb0e girl going into wooden building
		1000268201_693b08cb0e little girl climbing into wooden playhouse
		1000268201_693b08cb0e little girl climbing the stairs to her playhouse
		1000268201_693b08cb0e little girl in pink dress going into wooden cabin
'''


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


def loadValData():
    val_image_ids = load_set(trainConfig['directoryDev'])
    # Cargar descripciones
    val_captions = load_clean_descriptions('descriptions.txt', val_image_ids)
    # Cargar características de la imagen
    val_features = load_photo_features('features.pkl', val_image_ids)
    print('Imágenes disponibles para validación: ', len(val_features))
    return val_features, val_captions


'''
	- Los subtítulos deberán codificarse en números antes de presentarlos al modelo.
	- El primer paso para codificar los subtítulos es crear una correspondencia coherente entre palabras y valores enteros únicos.
		Keras proporciona la clase Tokenizer que puede aprender este mapeo a partir de los subtítulos cargados.
	- Adaptar un tokenizador a los subtítulos
'''


# Convertir el diccionario de descripciones en una lista
def to_lines(descriptions):
    all_desc = list()
    for key in descriptions.keys():
        [all_desc.append(caption) for caption in descriptions[key]]
    return all_desc


# Ajustar un token dadas las descripciones
def create_tokenizer(descriptions):
    lines = to_lines(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer


def create_sequences(tokenizer, max_length, descriptions, photos):
    X1, X2, y = list(), list(), list()
    vocab_size = len(tokenizer.word_index) + 1
    # Recorrer cada descripción de la imagen
    for desc in descriptions:
        # Codificar la secuencia
        seq = tokenizer.texts_to_sequences([desc])[0]
        # Dividir una secuencia en varios pares X,y
        for i in range(1, len(seq)):
            # Dividir en pares de entrada y salida
            in_seq, out_seq = seq[:i], seq[i]
            # Secuencia de entrada de relleno
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            # Codificar la secuencia de salida
            out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
            # Almacenamiento
            X1.append(photos)
            X2.append(in_seq)
            y.append(out_seq)
    return array(X1), array(X2), array(y)


# Generador de datos, destinado a ser utilizado en una llamada a model.fit_generator()
def data_generator(images, captions, tokenizer, max_length, batch_size, random_seed):
    # Configurar e valor "random seed" para variar los resultados. Inicializa el generador de números aleatorios
    random.seed(random_seed)
    # ID de imagen
    image_ids = list(captions.keys())
    _count = 0
    assert batch_size <= len(image_ids), 'El tamaño del lote debe ser inferior o igual a{}'.format(len(image_ids))
    while True:
        if _count >= len(image_ids):
            # El generador excedió o llegó al final y hay que reiniciar
            _count = 0
        # Lista de lotes para almacenar datos
        input_img_batch, input_sequence_batch, output_word_batch = list(), list(), list()
        for i in range(_count, min(len(image_ids), _count + batch_size)):
            # Recuperar el id de la imagen
            image_id = image_ids[i]
            # Recuperar el id de la imagen
            image = images[image_id][0]
            # Recuperar la lista de subtítulos
            captions_list = captions[image_id]
            # Lista aleatoria de subtítulos
            random.shuffle(captions_list)
            input_img, input_sequence, output_word = create_sequences(tokenizer, max_length, captions_list, image)
            # Añadir al lote
            for j in range(len(input_img)):
                input_img_batch.append(input_img[j])
                input_sequence_batch.append(input_sequence[j])
                output_word_batch.append(output_word[j])
        _count = _count + batch_size
        yield [[np.array(input_img_batch), np.array(input_sequence_batch)], np.array(output_word_batch)]


# Calcular la longitud de los pies de foto con más palabras
def max_length(descriptions):
    lines = to_lines(descriptions)
    return max(len(line.split()) for line in lines)


# Definir el modelo RNN
# La siguiente función denominada define_RNNModel() define y devuelve el
# modelo listo para ajustarse.

def define_RNNModel(vocab_size, max_len, trainConfig, model_type):
    embedding_size = trainConfig['embedding_size']
    # modelo extractor de características
    if model_type == 'inceptionv3':
        # InceptionV3 genera un vector de 2048 dimensiones para cada imagen, que introduciremos en el modelo RNN
        inputs1 = Input(shape=(2048,))
    elif model_type == 'vgg16':
        # VGG16 produce un vector de 4096 dimensiones para cada imagen, que alimentaremos al modelo RNN
        inputs1 = Input(shape=(4096,))
    elif model_type == 'resnet50':
        # ResNet50 genera un vector de 2048 dimensiones para cada imagen, que introduciremos en el modelo RNN
        inputs1 = Input(shape=(2048,))
    fe1 = Dropout(trainConfig['dropout'])(inputs1)
    fe2 = Dense(embedding_size, activation='relu')(fe1)

    # modelo procesador de secuencia
    inputs2 = Input(shape=(max_len,))
    se1 = Embedding(vocab_size, embedding_size, mask_zero=True)(inputs2)
    se2 = Dropout(trainConfig['dropout'])(se1)
    se3 = LSTM(trainConfig['LSTM_units'])(se2)

    # Modelo decodificador
    # Fusión de los modelos y creación de un clasificador softmax

    decoder1 = add([fe2, se3])
    decoder2 = Dense(trainConfig['dense_units'], activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)

    # Juntar [imagen, secuencia] [palabra]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Resumen del modelo
    print(model.summary())
    # plot_model(model, to_file="RNN_model.png", show_shapes=True)
    return model


# Definición del modelo RNN con diferente arquitectura

def define_AlternativeRNNModel(vocab_size, max_len, trainConfig, model_type):
    embedding_size = trainConfig['embedding_size']
    # modelo extractor de características
    if model_type == 'inceptionv3':
        # InceptionV3 genera un vector de 2048 dimensiones para cada imagen, que introduciremos en el modelo RNN
        image_input = Input(shape=(2048,))
    elif model_type == 'vgg16':
        # VGG16 produce un vector de 4096 dimensiones para cada imagen, que alimentaremos al modelo RNN
        image_input = Input(shape=(4096,))
    elif model_type == 'resnet50':
        # ResNet50 genera un vector de 2048 dimensiones para cada imagen, que introduciremos en el modelo RNN
        image_input = Input(shape=(2048,))
    image_model1 = Dense(embedding_size, activation='relu')(image_input)
    image_model = RepeatVector(max_len)(image_model1)

    caption_input = Input(shape=(max_len,))
    # mask_zero: Ponemos a cero las entradas con la misma longitud, la máscara cero ignora esas entradas
    lang_model1 = Embedding(vocab_size, embedding_size, mask_zero=True)(caption_input)
    # Predice la siguiente palabra usando las palabras anteriores
    # tenemos que establecer return_sequences = True.
    lang_model2 = LSTM(trainConfig['LSTM_units'], return_sequences=True)(lang_model1)
    lang_model3 = TimeDistributed(Dense(embedding_size))(lang_model2)

    # Se Fusionan los modelos y se crea un clasificador softmax
    final_model_1 = concatenate([image_model, lang_model3])
    final_model_2 = Bidirectional(LSTM(trainConfig['LSTM_units'], return_sequences=False))(final_model_1)
    final_model_3 = Dense(vocab_size, activation='relu')(final_model_2)
    final_model = Dense(vocab_size, activation='softmax')(final_model_3)

    model = Model(inputs=[image_input, caption_input], outputs=final_model)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
    # Resumen del modelo
    print(model.summary())
    # plot_model(model, to_file="alternativoRNN_model.png", show_shapes=True)
    return model


# Asignar un número entero a una palabra
def int_to_word(integer, tokenizer):
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
    intermediate_caption = [int_to_word(i, tokenizer) for i in start_word]

    final_caption = []

    for word in intermediate_caption:
        if word == 'endseq':
            break
        else:
            final_caption.append(word)

    final_caption.append('endseq')
    return ' '.join(final_caption)


# Validar con los datos de prueba
def evaluate_model_beam_search(model, images, captions, tokenizer, max_length, beam_index=3):
    actual, predicted = list(), list()
    for image_id, caption_list in tqdm(captions.items()):
        # Predecir el título de la imagen
        yhat = generate_caption_beam_search(model, tokenizer, images[image_id], max_length, beam_index=beam_index)
        # Dividir en palabras
        ground_truth = [caption.split() for caption in caption_list]
        # Agregar a la lista
        actual.append(ground_truth)
        predicted.append(yhat.split())
    # Calcular la puntuación BLEU
    print('Puntuación BLEU :')
    print(
        'Una coincidencia perfecta da como resultado una puntuación de 1,0, mientras que una falta de coincidencia perfecta da como resultado una puntuación de 0,0.')
    print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0)))
    print('BLEU-4: %f' % corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25)))


# LLAMADA DE METODOS
# Primera parte: extraer características de todas las imágenes
# Pimera parte: # Extraer las características de todas las imágenes
directoryDataset = trainConfig['directoryDataset']
model_type = trainConfig['model_type']
features = extract_features(directoryDataset, model_type)
print('Características extraídas: %d' % len(features))
# Guardar el archivo
dump(features, open('features.pkl', 'wb'))

# Segunda parte
directoryToken = trainConfig['directoryToken']
# Cargar las descripciones
descriptions = load_description(directoryToken)
print('Cargado: %d ' % len(descriptions))
# Limpiar las descripciones
clean_descriptions(descriptions)
# Sintetizar vocabulario
vocabulary = to_vocabulary(descriptions)
print('Tamaño del vocabulario: %d' % len(vocabulary))
# Guardar el archivo
save_descriptions(descriptions, 'descriptions.txt')

# Tercera parte: Dataset de entrenamiento

# Cargar el dataset entrenamiento (6K)
directoryTrain = trainConfig['directoryTrain']
train = load_set(directoryTrain)
print('Dataset: %d' % len(train))
# Descripciones
train_descriptions = load_clean_descriptions('descriptions.txt', train)
print('Descripciones: entrenamiento=%d' % len(train_descriptions))
# Características de las fotos
train_features = load_photo_features('features.pkl', train)
print('Fotos: entrenamiento=%d' % len(train_features))
# Preparar el tokenizer y guardarlo
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print('Tamaño del vocabulario: %d' % vocab_size)
# Determinar la longitud máxima de la secuencia
max_length = max_length(train_descriptions)
print('Longitud de Descripción : %d' % max_length)

# PARA ENTRENAMIENTO
# Cargar subtítulos
# Cargar características de la imagen

print('Imágenes disponibles para entrenamiento: ', len(train_features))

val_features, val_descriptions = loadValData()

# Ahora que tenemos las características de la imagen del modelo CNN AlternativeRNNModel,
# necesitamos alimentarlas a un modelo RNN.

# Definición del modelo RNN (RNNModel y AlternativeRNNModel)

# model = define_RNNModel(vocab_size, max_length, trainConfig, model_type)
model = define_AlternativeRNNModel(vocab_size, max_length, trainConfig, model_type)
print('Modelo RNN (decodificador) Resumen : ')
print(model.summary())

## Entrenar el modelo y guardar después de cada época

num_of_epochs = trainConfig['num_of_epochs']
batch_size = trainConfig['batch_size']
train_length = len(train_descriptions)
val_length = len(val_descriptions)
steps_train = train_length // batch_size
if train_length % batch_size != 0:
    steps_train = steps_train + 1
steps_val = val_length // batch_size
if val_length % batch_size != 0:
    steps_val = steps_val + 1

# Establecer "random_seed"  para la reproducibilidad de los resultados.
random_seed = trainConfig['random_seed']
# Mezclar los datos de entrenamiento
ids_train = list(train_descriptions.keys())
random.shuffle(ids_train)
train_descriptions = {_id: train_descriptions[_id] for _id in ids_train}

# Crear el generador de datos para el entrenamiento
# Devuelve [[img_features, text_features], out_word]
generator_train = data_generator(train_features, train_descriptions, tokenizer, max_length, batch_size, random_seed)
# Crear el generador de datos para la validación
generator_val = data_generator(val_features, val_descriptions, tokenizer, max_length, batch_size, random_seed)

model_save_path = "model_" + model_type + "_epoch-{epoch:02d}_train_loss-{loss:.4f}_val_loss-{val_loss:.4f}.hdf5"
# Definición del "callback" para el punto de control
checkpoint = ModelCheckpoint(model_save_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks = [checkpoint]

print('Pasos_entrenamiento: {}, pasos_val: {}'.format(steps_train, steps_val))
print('Batch Size: {}'.format(batch_size))
print('Número total de épocas = {}'.format(num_of_epochs))

model.fit_generator(generator_train,
                    epochs=num_of_epochs,
                    steps_per_epoch=steps_train,
                    validation_data=generator_val,
                    validation_steps=steps_val,
                    callbacks=callbacks,
                    verbose=1)

# Evalar el modelo con los datos de validación y obtener la puntuación BLEU
print('Modelo entrenado con éxito.')
print(
    'Ejecución del modelo en el conjunto de validación para calcular la puntuación BLEU utilizando BEAM search con k={}'.format(
        '3'))
evaluate_model_beam_search(model, val_features, val_descriptions, tokenizer, max_length, beam_index=3)
