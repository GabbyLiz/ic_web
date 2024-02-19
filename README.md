## Reconocimiento de objetos y semántica de una imagen con Image Captioning

El presente repositorio contiene los archivos necesarios para entrenar una red capaz de describir el contenido de una imagen y desplegarla en un navegador web. Esta técnica utiliza el Procesamiento del Lenguaje Natural y la Visión por Computador para generar las descripciones.

## Información sobre el proyecto

Para el entrenamiento se han utilizado 2 datasets: Flickr 8k y Coco. En el siguiente enlace pueden descargarse ambos datasets con los archivos necesarios para su uso: 

<strong>Datasets:</strong> <a href="https://uceedu-my.sharepoint.com/:u:/g/personal/glquinde_uce_edu_ec/Ee_62RHxI4lIkM3HH5-OY1kBbitnbu9ccapGp16INjfDVQ?e=lnXHNn">enlace de descarga</a>

Para ejecutar el proyecto realice los siguientes pasos:

1. Clonar el repositorio.<br>
2. Crear un ambiente para el proyecto, de esta manera no creará conflicto entre librerías previamente instaladas. En el archivo `installation_commands.txt` encontrará los pasos a realizar.
3. En la carpeta`train_val_data` colocar los archivos descargados de los datasets.
4. Con el archivo `train_modify.py` se realizarán los entrenamientos con su respectiva evaluación BLEU.
5. Con el archivo `test.py` evaluaremos los entrenamientos en una cantidad de imágenes determinada alojada en las carpetas .
6. Con el archivo `app.py` desplegará una aplicacion web de manera local donde podrá interactuar con los mejores modelos entrenados en este proyecto,  están alojados en la carpeta `model_data`.

Es importante que cuente con una computadora con al menos 8GB de RAM y si dispone de una GPU los entrenamientos serán más precisos.



## Referencias

<ul type="square">
	<li><a href="https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf">Show and Tell: A Neural Image Caption Generator</a> - Oriol Vinyals, Alexander Toshev, Samy Bengio, Dumitru Erhan</li>
	<li><a href="https://machinelearningmastery.com/develop-a-deep-learning-caption-generation-model-in-python/">How to Develop a Deep Learning Photo Caption Generator from Scratch</a> - Jason Brownlee</li>
	<li><a href="https://github.com/dabasajay/Image-Caption-Generator/tree/master">Image Caption Generator</a> - Ajay Dabas</li>
	<li><a href="https://yashk2810.github.io/Image-Captioning-using-InceptionV3-and-Beam-Search/">Image-Captioning using InceptionV3 and Beam Search</a> - Yash Katariya</li>
	<li><a href="https://github.com/Shobhit20/Image-Captioning/tree/master">Image Captioning: Implementing the Neural Image Caption Generator with python</a> - Shobhit Maheshwari</li>
	
</ul>
