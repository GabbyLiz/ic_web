from os import listdir
from tqdm import tqdm
from numpy import array
from pickle import load, dump
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical, plot_model
from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout
from keras.layers.merge import add
import os
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
    #Extraer descripciones de las imagenes
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
        count = count+1
    print('{}: Subtítulos procesados: {}',count)
    return mapping

"""
	Limpieza del texto de las descripciones
	Dado el diccionario de identificadores de imagen a las descripciones
	recorre cada descripcion y limpia el texto
"""
# Preparar la tabla de traducción para eliminar los signos de puntuación

def clean_descriptions(descriptions):

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
            desc = [word for word in desc if len(word)>1]
            # Eliminar tokens con números
            desc = [word for word in desc if word.isalpha()]
            # Guardar como cadena
            desc_list[i] =  ' '.join(desc)


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
def save_captions(descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

####

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
    #Cerrar el archivo
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
def load_clean_description(filename, dataset):
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
def load_image_features(filename, dataset):
    # Cargar todas las características
    all_features = load(open(filename, 'rb'))
    # Características del filtro
    features = {id: all_features[id] for id in dataset}
    return features

'''
	- Los subtítulos deberán codificarse en números antes de presentarlos al modelo.
	- El primer paso para codificar los subtítulos es crear una correspondencia coherente entre palabras y valores enteros únicos.
		Keras proporciona la clase Tokenizer que puede aprender este mapeo a partir de los subtítulos cargados.
	- Adaptar un tokenizador a los subtítulos
'''

# Convertir el diccionario de descripciones en una lista
def to_lines(descriptions):
    all_captions = list()
    for key_image_id in descriptions.keys():
        [all_captions.append(caption) for caption in descriptions[key_image_id]]
    return all_captions

# Ajustar un token dadas las descripciones
def create_tokenizer(train_descriptions):
    lines = to_lines(train_descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

def create_sequences(tokenizer, max_length, desc_list, photos):
    X1, X2, y = list(), list(), list()
    vocab_size = len(tokenizer.word_index) + 1
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
        X1.append(photos)
        X2.append(in_seq)
        y.append(out_seq)
    return array(X1), array(X2), array(y)


# generador de datos, destinado a ser utilizado para model.fit_generator()
def data_generator(descriptions, photos, tokenizer, max_length):
    while 1:
        for key, desc_list in descriptions.items():
            # Recuperar las características de la foto
            photo = photos[key][0]
            in_img, in_seq, out_word = create_sequences(tokenizer, max_length, desc_list, photo)
            yield [in_img, in_seq], out_word

# Calcular la longitud de los pies de foto con más palabras
def calc_max_length(captions):
    lines = to_lines(captions)
    return max(len(line.split()) for line in lines)


def define_model(vocab_size, max_len):
    # Modelo extractor de características
    # VGG16 produce un vector de 4096 dimensiones para cada imagen, que alimentaremos al modelo RNN
    inputs1 = Input(shape=(4096,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    # Modelo procesador de secuencia
    inputs2 = Input(shape=(max_len,))
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
    #model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Resumen del modelo
    print(model.summary())
    plot_model(model, to_file="RNN_model.png", show_shapes=True)
    return model



##### LLAMADA DE METODOS
#train autor1
# # extract features from all images
# directoryDataset = 'train_val_data/Flickr8k_Dataset/'
# features = extract_features3(directoryDataset, 'vgg16')
# print('Extracted Features: %d' % len(features))
# # save to file
# dump(features, open('features.pkl', 'wb'))
#
# #train_autor2
# directoryToken = 'train_val_data/Flickr8k.token.txt'
# # load descriptions
# descriptions = load_description(directoryToken)
# print('Loaded: %d ' % len(descriptions))
# # clean descriptions
# clean_descriptions(descriptions)
# # summarize vocabulary
# vocabulary = to_vocabulary(descriptions)
# print('Vocabulary Size: %d' % len(vocabulary))
# # save to file
# save_captions(descriptions, 'descriptions.txt')

#train_autor3

# Cargar conjunto de datos de entrenamiento (6K)
directoryTrain = 'train_val_data/Flickr_8k.trainImages.txt'
train = load_set(directoryTrain)
# Descripciones
train_descriptions = load_clean_description('descriptions.txt', train)
print('Descriptions: train=%d' % len(train_descriptions))
# Características de las fotos
train_features = load_image_features('features.pkl', train)
print('Photos: train=%d' % len(train_features))

# Preparar el tokenizador y guardarlo
tokenizer = create_tokenizer(train_descriptions)
dump(tokenizer, open('tokenizer.pkl', 'wb'))
vocab_size = len(tokenizer.word_index) + 1
print('Vocabulary Size: %d' % vocab_size)


# Determina la longitud máxima de la secuencia
max_length = calc_max_length(train_descriptions)
print('Description Length: %d' % max_length)

# Generador de datos para prueba
generator = data_generator(train_descriptions, train_features, tokenizer, max_length)


inputs, outputs = next(generator)
print(inputs[0].shape)
print(inputs[1].shape)
print(outputs.shape)

# Entrenamiento
model = define_model(vocab_size, max_length)
# Entreanr el modelo, ejecutar las épocas manualmente y guardar el modelo con cada época
epochs = 20
steps = len(train_descriptions)
for i in range(epochs):
    # Creación del generador de datos
    generator = data_generator(train_descriptions, train_features, tokenizer, max_length)
    # Configurado para una época
    model.fit_generator(generator, epochs=1, steps_per_epoch=steps, verbose=1)
    # Guardar el modelo
    model.save('model_' + str(i) + '.h5')