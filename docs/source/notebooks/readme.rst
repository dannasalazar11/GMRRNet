GMRRNet
=======

Instalar librería y modelo
--------------------------

Para poder tener acceso al software basta con tener la siguiente linea
dentro de tu cuaderno

.. code:: ipython3

    !pip install -U git+https://github.com/dannasalazar11/GMRRNet

Una vez instalada, se podrá llamar la clase con la siguiente linea

.. code:: ipython3

    from gmrrnet.model import GMRRNet

Entrenamiento y testeo
----------------------

El entrenamiento se realiza como para cualquier modelo de keras
(teniendo en cuenta que ya hay datos de entrenamiento y de test). Para
ver el ejemplo completo con la base de datos GIGA_MI_ME acceda al
ejemplo en
GMRRNet:raw-latex:`\docs`:raw-latex:`\source`:raw-latex:`\notebooks`\\01-example.ipynb

.. code:: ipython3

    import tensorflow as tf
    
    model = GMRRNet()
    
    history = model.fit(X_train,y_train, epochs=150, batch_size=32, verbose=1)

from sklearn.metrics import accuracy_score y_pred =
model.predict(X_test)[0] accuracy_score(np.argmax(y_test, axis=1),
np.argmax(y_pred, axis=1))

Visualización e interpretabilidad
---------------------------------

Topoplots
~~~~~~~~~

.. code:: ipython3

    !pip install gmrrnet[dev]

.. code:: ipython3

    from gmrrnet.utils import topoplot
    from gmrrnet.utils import plot_circos
    import numpy as np
    import tensorflow as tf
    import matplotlib.pyplot as plt

kernel1 =
tf.keras.Model(inputs=model.inputs,outputs=model.get_layer(‘gaussian_layer_1’).output)
kernel2 =
tf.keras.Model(inputs=model.inputs,outputs=model.get_layer(‘gaussian_layer_2’).output)
kernel3 =
tf.keras.Model(inputs=model.inputs,outputs=model.get_layer(‘gaussian_layer_3’).output)

idx_left = tf.squeeze(tf.where(np.argmax(y_train, axis=1)==0)) idx_right
= tf.squeeze(tf.where(np.argmax(y_train, axis=1)==1))

kernel 1
~~~~~~~~

X_k1 = kernel1.predict(tf.expand_dims(X_train[0], axis=0)) X_k1 =
tf.reduce_mean(X_k1, axis=-1) # promedio por filtros

X_k1_left = tf.reduce_mean(tf.gather(X_k1, idx_left), axis=0) # promedio
de clase izq X_k1_right = tf.reduce_mean(tf.gather(X_k1, idx_right),
axis=0) # promedio de clase der

Kernel 2
~~~~~~~~

X_k2 = kernel2.predict(X_train) X_k2 = tf.reduce_mean(X_k2, axis=-1) #
promedio por filtros

X_k2_left = tf.reduce_mean(tf.gather(X_k2, idx_left), axis=0) # promedio
de clase izq X_k2_right = tf.reduce_mean(tf.gather(X_k2, idx_right),
axis=0) # promedio de clase der

Kernel 3
~~~~~~~~

X_k3 = kernel3.predict(X_train) X_k3 = tf.reduce_mean(X_k3, axis=-1) #
promedio por filtros

X_k3_left = tf.reduce_mean(tf.gather(X_k3, idx_left), axis=0) # promedio
de clase izq X_k3_right = tf.reduce_mean(tf.gather(X_k3, idx_right),
axis=0) # promedio de clase der

.. code:: ipython3

    fig, axs = plt.subplots(3,1,figsize=[40,10])
    
    axs[0].set_title("Differences")
    
    axs[0].set_ylabel("$\sigma=0.8$")
    axs[1].set_ylabel("$\sigma=2.2$")
    axs[2].set_ylabel("$\sigma=4.8$")
    
    # diferencias
    diferencia1 = tf.abs(tf.subtract(tf.reduce_mean(X_k1_left,axis=0).numpy() , tf.reduce_mean(X_k1_right,axis=0).numpy()))
    diferencia2 = tf.abs(tf.subtract(tf.reduce_mean(X_k2_left,axis=0).numpy() , tf.reduce_mean(X_k2_right,axis=0).numpy()))
    diferencia3 = tf.abs(tf.subtract(tf.reduce_mean(X_k3_left,axis=0).numpy() , tf.reduce_mean(X_k3_right,axis=0).numpy()))
    max_dif = tf.reduce_max(tf.stack([diferencia1, diferencia2, diferencia3]))
    
    vmax = tf.reduce_max(tf.stack([tf.reduce_mean(X_k1_left,axis=0),tf.reduce_mean(X_k1_right,axis=0),tf.reduce_mean(X_k2_left,axis=0),tf.reduce_mean(X_k2_right,axis=0),tf.reduce_mean(X_k3_left,axis=0),tf.reduce_mean(X_k3_right,axis=0)], axis=0)) 
    
    topoplot(diferencia1, eeg_ch_names, contours=3, names=eeg_ch_names, sensors=False, ax= axs[0], vlim=(0,max_dif))
    topoplot(diferencia2, eeg_ch_names, contours=3, names=eeg_ch_names, sensors=False, ax=axs[1], vlim=(0,max_dif))
    topoplot(diferencia3, eeg_ch_names, contours=3, names=eeg_ch_names, sensors=False, ax=axs[2], vlim=(0,max_dif))
    
    plt.savefig('heads_43.pdf', bbox_inches='tight')
    plt.show()

Conectividad circos
~~~~~~~~~~~~~~~~~~~

.. code:: ipython3

    areas = {
        'Frontal': ['Fpz', 'AFz', 'Fz', 'FCz'],
        'Frontal Right': ['Fp2','AF4','AF8','F2','F4','F6','F8',],
        'Central Right': ['FC2','FC4','FC6','FT8','C2','C4','C6','T8','CP2','CP4','CP6','TP8',],
        'Posterior Right': ['P2','P4','P6','P8','P10','PO4','PO8','O2',],
        #'Central': ['Cz'],
        'Posterior': ['CPz','Pz', 'Cz','POz','Oz','Iz',],
        'Posterior Left': ['P1','P3','P5','P7','P9','PO3','PO7','O1',],
        'Central Left': ['FC1','FC3','FC5','FT7','C1','C3','C5','T7','CP1','CP3','CP5','TP7',],
        'Frontal Left': ['Fp1','AF3','AF7','F1','F3','F5','F7',],
    }
    
    plot_circos(X_k2_left, eeg_ch_names, areas, threshold=0.75)
