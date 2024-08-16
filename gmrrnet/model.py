import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    Flatten,
    Dense,
    Activation,
    concatenate,
    Layer,
    Conv2D,
    BatchNormalization,
    Lambda,
)
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.losses import Loss


def renyi_entropy(K, alpha=2):
    """
    Calculates the Rényi entropy for an input tensor.

    Parameters:
    -----------
    K : Tensor
        An input tensor of shape `(N, F, C, C)`, where:
        - N: Number of samples in the batch.
        - F: Number of filters or features.
        - C: Number of channels or dimensions of the square matrices within the tensor.

    alpha : float, optional
        The Rényi entropy parameter. Default is 2.0.
        - When `alpha=2`, a specific optimization is applied for this value.

    Returns:
    --------
    Tensor
        An output tensor of shape `(N, F)`, which contains the calculated Rényi entropy for each sample and filter.

    Description:
    ------------
    Rényi entropy is a generalization of Shannon entropy, which depends on the parameter `alpha`. This function calculates the Rényi entropy for each of the normalized square matrices in the tensor `K`.

    Steps:
    ------
    1. **Kernel Normalization**:
        - The tensor `K` is normalized by dividing it by the product of the diagonal elements of its square matrices.
        - This normalization is performed to stabilize the entropy calculation.

    2. **Entropy Calculation**:
        - If `alpha=2`, an optimization is used that computes the trace of the matrix product of `X` with itself.
        - For other values of `alpha`, the eigenvalues of `X` are computed, and the general formula for Rényi entropy is applied.

    Usage Example:
    --------------
    ```python
    # Create a sample tensor with shape (N, F, C, C)
    K = tf.random.normal((32, 10, 64, 64))

    # Calculate the Rényi entropy with alpha = 2
    entropy = renyi_entropy(K, alpha=2)
    print(entropy.shape)  # Output: (32, 10)

    """

    # Obtener el número de canales
    C = K.shape[-1]

    # Normalizamos el kernel antes de calcular la entropía

    # Crear una máscara para obtener los elementos diagonales
    diag = tf.expand_dims(tf.linalg.diag_part(K), -1)
    # Calcular el producto de los elementos diagonales
    denominator = tf.math.sqrt(
        tf.linalg.matmul(diag, diag, transpose_b=True)
    )
    # Normalización

    X = (1 / C) * tf.math.divide(K, denominator)
    if alpha == 2:
        # Realiza el producto matricial entre las dos últimas dimensiones
        X_matmul = tf.linalg.matmul(X, X)
        return -tf.math.log(tf.linalg.trace(X_matmul))
    else:
        # Calcula los autovalores y autovectores de las dos últimas dimensiones
        e, _ = tf.linalg.eigh(X)
        # Calcula la entropía de Renyi
        return tf.math.log(
            tf.reduce_sum(tf.math.real(tf.math.pow(e, alpha)), axis=-1)
        ) / (1 - alpha)


def joint_renyi_entropy(K, alpha):
    """
    Calculates the joint Rényi entropy for an input tensor.

    Parameters:
    -----------
    K : Tensor
        An input tensor of shape `(N, F, C, C)`, where:
        - N: Number of samples in the batch.
        - F: Number of filters or features.
        - C: Number of channels or dimensions of the square matrices within the tensor.

    alpha : float
        The Rényi entropy parameter. Controls the sensitivity of the metric to different probability distributions.

    Returns:
    --------
    Tensor
        An output tensor of shape `(N, 1)`, which contains the calculated joint Rényi entropy for each sample.

    Description:
    ------------
    This function calculates the joint Rényi entropy, which measures the amount of uncertainty in a system by considering multiple variables together. Joint entropy is useful for evaluating the dependency or interrelation between variables.

    Steps:
    ------
    1. **Product of Tensor Terms**:
        - A product is performed along the `F` dimension of the tensor `K`, reducing the tensor from `(N, F, C, C)` to `(N, C, C)`.

    2. **Trace Calculation**:
        - The trace of the resulting tensor is calculated, which is the sum of the elements on the main diagonal of the square matrices in `(N, C, C)`.
        - The trace is expanded and repeated to make it compatible with the dimensionality of tensor `K`.

    3. **Normalization**:
        - The product obtained in the first step is normalized by dividing it by the expanded trace, stabilizing the entropy calculation.

    4. **Joint Entropy Calculation**:
        - The normalized tensor is passed to the `renyi_entropy` function, which calculates the Rényi entropy while considering the interrelation between the variables.
    """


    # Obtener el número de canales
    C = K.shape[-1]

    # Producto de los términos a lo largo de la dimensión F
    product = tf.reduce_prod(K, axis=1)  # (N, C, C)

    # Calcular la traza de las matrices cuadradas
    trace = tf.linalg.trace(product)
    trace = tf.expand_dims(tf.expand_dims(trace, axis=-1), axis=-1)
    trace = tf.tile(trace, [1, C, C])

    # Normalizar el producto
    argument = product / trace
    argument = tf.expand_dims(
        argument, axis=1
    )  # Se necesita porque renyi_entropy recibe 4 dimensiones (1, C, C)

    # Calcular la entropía conjunta usando la función renyi_entropy
    joint_entropy = renyi_entropy(argument, alpha=alpha)

    return joint_entropy


def inception_block(x, filters, sigmas):
    """
    Builds a custom Inception block that includes convolution layers and Gaussian kernel layers.

    This Inception block creates three branches, each applying a Gaussian kernel followed by a 2D convolution layer.
    Finally, the outputs of these branches are concatenated along the channel axis.

    Parameters:
    -----------
    x : Tensor
        Input tensor of shape `(N, C, T, F)` where:
        - N: Number of samples in the batch.
        - C: Number of channels or features.
        - T: Number of time steps.
        - F: Number of additional filters or features.

    filters : list of int
        List containing the number of filters for each branch of the Inception block. It should be a list of three integers `[f1, f2, f3]`
        where `f1`, `f2`, and `f3` are the number of filters for branches 1, 2, and 3, respectively.

    sigmas : list of float
        List containing the `sigma` values for each `GaussianKernelLayer` in the branches of the Inception block. It should be a list
        of three values `[sigma1, sigma2, sigma3]` where `sigma1`, `sigma2`, and `sigma3` correspond to branches 1, 2, and 3, respectively.

    Returns:
    --------
    branch_k1, branch_k2, branch_k3 : Tensors
        The outputs of the `GaussianKernelLayer` in the three branches, with shape `(N, C, C, F)`.

    output : Tensor
        The concatenated output of the three branches after the convolution layer, with shape `(N, C, T, f1 + f2 + f3)`.

    Description:
    ------------
    This Inception block consists of the following parts:
    1. **Branch 1**:
        - Applies a `GaussianKernelLayer` with `sigma=sigmas[0]` to the input `x`.
        - Applies a 2D convolution layer with `f1` filters of size `(3, 3)` and `ReLU` activation.

    2. **Branch 2**:
        - Applies a `GaussianKernelLayer` with `sigma=sigmas[1]` to the input `x`.
        - Applies a 2D convolution layer with `f2` filters of size `(3, 3)` and `ReLU` activation.

    3. **Branch 3**:
        - Applies a `GaussianKernelLayer` with `sigma=sigmas[2]` to the input `x`.
        - Applies a 2D convolution layer with `f3` filters of size `(3, 3)` and `ReLU` activation.

    Finally, the outputs of the three branches are concatenated along the channel axis.
    """


    # Filtros
    f1, f2, f3 = filters

    # Rama 1: Aplicar el kernel Gaussiano seguido de una convolución 2D
    branch_k1 = GaussianKernelLayer(
        sigma=sigmas[0], name="gaussian_layer_1"
    )(x)
    branch1 = Conv2D(f1, (3, 3), padding='same', activation='relu')(
        branch_k1
    )

    # Rama 2: Aplicar el kernel Gaussiano seguido de una convolución 2D
    branch_k2 = GaussianKernelLayer(
        sigma=sigmas[1], name="gaussian_layer_2"
    )(x)
    branch2 = Conv2D(f2, (3, 3), padding='same', activation='relu')(
        branch_k2
    )

    # Rama 3: Aplicar el kernel Gaussiano seguido de una convolución 2D
    branch_k3 = GaussianKernelLayer(
        sigma=sigmas[2], name="gaussian_layer_3"
    )(x)
    branch3 = Conv2D(f3, (3, 3), padding='same', activation='relu')(
        branch_k3
    )

    # Concatenar las salidas de las tres ramas a lo largo del eje de los canales
    output = concatenate([branch1, branch2, branch3], axis=-1)

    return branch_k1, branch_k2, branch_k3, output


class GaussianKernelLayer(Layer):
    """
    Custom Keras layer that applies a Gaussian kernel to the inputs.

    This layer calculates the squared Euclidean distance between pairs of points and then applies a Gaussian kernel function
    to transform those distances into similarities, which is useful in signal processing such as EEG or in constructing neural networks with kernel functions.

    Parameters:
    -----------
    sigma : float, optional (default=1.0)
        Standard deviation of the Gaussian kernel function. Controls the range or "spread" of the Gaussian.


    """


    def __init__(self, sigma=1.0, **kwargs):
        super(GaussianKernelLayer, self).__init__(**kwargs)
        self.sigma = sigma

    def build(self, input_shape):
        """
                Initializes the layer. This method is called once and is used to build the layer's variables.

                Parameters:
                -----------
                input_shape : tuple
                    The expected shape of the input to the layer.
        """

        super(GaussianKernelLayer, self).build(input_shape)

    def call(self, inputs):
        """
                Applies the Gaussian kernel transformation to the input data.

                Parameters:
                -----------
                inputs : Tensor
                    Input tensor of shape `(N, C, T, F)`.

                Returns:
                --------
                gaussian_kernel : Tensor
                    Output tensor where the Gaussian kernel has been applied, with shape `(N, C, C, F)`.
        """

        # Descomposición de la forma del tensor de entrada
        N, C, T, F = (
            tf.shape(inputs)[0],
            tf.shape(inputs)[1],
            tf.shape(inputs)[2],
            tf.shape(inputs)[3],
        )

        # Reorganizar el tensor de entrada a la forma (N*F, C, T)
        inputs = tf.transpose(
            inputs, perm=(0, 3, 1, 2)
        )  # Cambia la forma a (N, F, C, T)
        inputs_reshaped = tf.reshape(inputs, (N * F, C, T))

        # Calcular la distancia euclidiana al cuadrado entre pares de puntos
        squared_differences = tf.expand_dims(
            inputs_reshaped, axis=2
        ) - tf.expand_dims(
            inputs_reshaped, axis=1
        )  # (N*F, C, C, T)
        squared_differences = tf.square(
            squared_differences
        )  # (N*F, C, C, T)
        pairwise_distances_squared = tf.reduce_sum(
            squared_differences, axis=-1
        )  # (N*F, C, C)
        pairwise_distances_squared = tf.reshape(
            pairwise_distances_squared, (N, F, C, C)
        )  # (N, F, C, C)
        pairwise_distances_squared = tf.transpose(
            pairwise_distances_squared, perm=(0, 2, 3, 1)
        )  # (N, C, C, F)

        # Calcular el kernel Gaussiano
        gaussian_kernel = tf.exp(
            -pairwise_distances_squared / (2.0 * tf.square(self.sigma))
        )

        return gaussian_kernel


class RenyiMutualInformation(Loss):
    def __init__(self, C, **kwargs):
        self.C = C
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
                y_true:
                y_pred: N x (F+1) where F are the marginal entropies and the last value is the joint entropy.
        """


        F = y_pred.shape[1] - 1
        entropy, joint_entropy = tf.split(y_pred, [F, 1], axis=-1)

        # Cast todo
        entropy = tf.cast(entropy, tf.float64)
        joint_entropy = tf.cast(joint_entropy, tf.float64)
        log_C = tf.math.log(tf.cast(self.C, tf.float64))

        mutual_information = tf.math.abs(
            (
                tf.expand_dims(tf.reduce_sum(entropy, axis=-1), axis=-1)
                - joint_entropy
            )
        ) / (
            F * log_C
        )  # normalizado

        return mutual_information


class RenyiMutualInformation(Loss):
    def __init__(self, C, **kwargs):
        """
        Initializes the RenyiMutualInformation class.

        Parameters:
        -----------
        C : int or float
            The number of channels (dimension C) used for normalization in the mutual information calculation.
        kwargs : dict
            Other optional arguments passed to the base `Loss` class.
        """

        self.C = C
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        Computes the loss based on Rényi mutual information.

        Parameters:
        -----------
        y_true : Tensor
            True labels, not used in this loss calculation but required to comply with the Keras API.

        y_pred : Tensor
            Prediction tensor of shape `(N, F+1)`, where:
            - N: Number of samples in the batch.
            - F: Number of marginal entropies.
            - F+1: The last column contains the joint entropy.

        Returns:
        --------
        Tensor
            A loss tensor of shape `(N, 1)`, which contains the calculated Rényi mutual information for each sample.

        Description:
        ------------
        This class implements a loss based on Rényi mutual information, which is a measure of the amount of information shared between two or more variables. In this case, it is used to evaluate how well the model predictions reflect the dependency between different features.

        Steps:
        ------
        1. **Separation of Entropies**:
            - The `y_pred` tensor is split into two parts: `entropy` containing the marginal entropies of each feature, and `joint_entropy` containing the joint entropy.

        2. **Type Casting**:
            - Ensures that both the marginal entropies and joint entropy are in `tf.float64` format for higher precision in calculations.

        3. **Calculation of the Logarithm of `C`**:
            - The natural logarithm of `C` (`log_C`) is computed, which is a constant value used in normalizing the calculation.

        4. **Calculation of Rényi Mutual Information**:
            - The sum of the marginal entropies is computed, the joint entropy is subtracted, and the result is normalized by `F * log_C` to obtain the normalized Rényi mutual information.

        Usage Example:
        ---------------
        ```python
        # Define the loss in the model
        model.compile(optimizer='adam', loss=[NormalizedBinaryCrossentropy(), RenyiMutualInformation(C=64)], loss_weights=[0.8, 0.2])
        ```

        Notes:
        ------
        - This class is designed to work in conjunction with a model that outputs both the marginal entropies and the joint entropy.
        - Rényi mutual information is useful in tasks where it is important to evaluate the amount of shared information between different features or signals.
        """


        # Número de entropías marginales
        F = y_pred.shape[1] - 1

        # Separar entropías marginales y entropía conjunta
        entropy, joint_entropy = tf.split(y_pred, [F, 1], axis=-1)

        # Convertir a tf.float64
        entropy = tf.cast(entropy, tf.float64)
        joint_entropy = tf.cast(joint_entropy, tf.float64)
        log_C = tf.math.log(tf.cast(self.C, tf.float64))

        # Calcular la información mutua de Rényi
        mutual_information = tf.math.abs(
            (
                tf.expand_dims(tf.reduce_sum(entropy, axis=-1), axis=-1)
                - joint_entropy
            )
        ) / (
            F * log_C
        )  # Normalización

        return mutual_information


class NormalizedBinaryCrossentropy(Loss):
    def __init__(self, **kwargs):
        """
                Initializes the NormalizedBinaryCrossentropy class.

                Parameters:
                -----------
                kwargs : dict
                    Optional arguments passed to the base `Loss` class.
        """
 
        super().__init__(**kwargs)

    def call(self, y_true, y_pred):
        """
        Computes the normalized binary cross-entropy loss.

        Parameters:
        -----------
        y_true : Tensor
            Tensor of true labels with shape `(N, 2)`, where:
            - N: Number of samples in the batch.
            - 2: Corresponds to binary classes (0 or 1).

        y_pred : Tensor
            Tensor of predictions with shape `(N, 2)`, where:
            - N: Number of samples in the batch.
            - 2: Probability predictions for the two binary classes.

        Returns:
        --------
        Tensor
            A loss tensor with shape `(N,)`, containing the normalized binary cross-entropy loss for each sample.

        Description:
        ------------
        This class implements a normalized version of the binary cross-entropy loss. Binary cross-entropy is a measure of dissimilarity between two probability distributions, commonly used as a loss function in binary classification problems.

        In this implementation, the binary cross-entropy loss is normalized using the theoretical losses associated with the labels `[1, 0]` and `[0, 1]`, which represent the two possible classes. The normalization aims to adjust the loss to be more robust against skewed probability distributions.

        Steps:
        ------
        1. **Calculation of Binary Cross-Entropy Loss**:
            - Computes the standard binary cross-entropy loss between `y_true` and `y_pred`.

        2. **Calculation of Theoretical Losses**:
            - Generates the theoretical losses `cce_left` and `cce_right` using the labels `[1.0, 0.0]` and `[0.0, 1.0]`, respectively.

        3. **Normalization**:
            - The original loss is divided by the sum of the theoretical losses, resulting in a normalized version of the loss.

        """

        # Obtener el tamaño del lote
        batch_size = tf.shape(y_pred)[0]
        batch_size_float = tf.cast(batch_size, tf.float32)

        # Calcular la entropía cruzada binaria estándar
        cce = tf.keras.losses.binary_crossentropy(y_true, y_pred)

        # Crear etiquetas teóricas para cada clase
        left = tf.tile(tf.expand_dims([1.0, 0.0], axis=0), [batch_size, 1])
        right = tf.tile(tf.expand_dims([0.0, 1.0], axis=0), [batch_size, 1])

        # Calcular la entropía cruzada binaria para las etiquetas teóricas
        cce_left = tf.keras.losses.binary_crossentropy(left, y_pred)
        cce_right = tf.keras.losses.binary_crossentropy(right, y_pred)

        # Normalizar la entropía cruzada binaria estándar
        cce_norm = tf.divide(cce, (cce_left + cce_right))

        return cce_norm


########################################################################
class GMRRNet:
    """"""

    # ----------------------------------------------------------------------
    def __new__(
        self,
        nb_classes=2,
        Chans=64,
        Samples=320,
        kernLength=64,
        norm_rate=0.25,
        alpha=2,
    ):
        """
        Builds a convolutional neural network model based on Inception blocks with convolutional layers and Rényi entropy computation.

        Parameters:
        -----------
        nb_classes : int, optional (default=2)
            Number of output classes for classification.

        Chans : int, optional (default=64)
            Number of input channels (spatial dimension).

        Samples : int, optional (default=320)
            Number of input samples (temporal dimension).

        kernLength : int, optional (default=64)
            Kernel length for the first convolutional layer.

        norm_rate : float, optional (default=0.25)
            Normalization rate for regularization in dense layers.

        alpha : int, optional (default=2)
            Order parameter for Rényi entropy, where alpha=2 represents quadratic Rényi entropy.

        Returns:
        --------
        model : tf.keras.Model
            The compiled neural network model ready for training.

        Description:
        ------------
        This function creates and compiles a convolutional neural network model with the following features:

        1. **Input**:
            - The input is a 4D tensor with shape `(Chans, Samples, 1)`, where `Chans` is the number of channels (spatial dimension) and `Samples` is the number of samples (temporal dimension).

        2. **First Convolutional Layer**:
            - A `Conv2D` layer with `F1=3` filters and a kernel of length `kernLength=64`.

        3. **Inception Block**:
            - A custom Inception block that applies a Gaussian filter with different sigmas and passes the results to a `Conv2D`.
            - Three different sigmas are used: `sigma1=0.8`, `sigma2=2.2`, and `sigma3=4.8`.
            - Three `Conv2D` layers are applied with `F2=5` filters each, and their outputs are concatenated.

        4. **Rényi Entropy Calculation**:
            - The outputs of the Gaussian kernel layers from the Inception block are concatenated, and the marginal and joint Rényi entropy are computed using the `renyi_entropy` and `joint_renyi_entropy` functions, respectively.

        5. **Final Layers**:
            - Another `Conv2D` layer with `F3=3` filters is added, followed by batch normalization (`BatchNormalization`), flattening (`Flatten`), and two dense layers with a `softmax` activation function for the final output.

        6. **Model Compilation**:
            - The model is compiled with the `Adam` optimizer.
            - A combination of two loss functions is used:
                1. `NormalizedBinaryCrossentropy`: A normalized binary cross-entropy loss.
                2. `RenyiMutualInformation`: A loss based on Rényi mutual information between the calculated entropies.
            - The losses are weighted with `loss_weights=[0.8, 0.2]`.
        """


        ###### Definición de los filtros para las capas convolucionales
        F1 = 3
        F2 = 5
        F3 = 3

        ###### Definición de la entrada
        input1 = Input(shape=(Chans, Samples, 1))

        ##################################################################
        # Primera capa convolucional con normalización por lotes
        conv2D = Conv2D(
            F1,
            (1, kernLength),
            padding='same',
            name='Conv2D_1',
            input_shape=(Chans, Samples, 1),
            use_bias=False,
        )(input1)
        block1 = BatchNormalization()(conv2D)

        # Definición de los sigmas para el bloque de Inception
        sigma1 = 0.8
        sigma2 = 2.2
        sigma3 = 4.8

        # Bloque de Inception
        branch_k1, branch_k2, branch_k3, inception = inception_block(
            block1, [F2, F2, F2], [sigma1, sigma2, sigma3]
        )

        ##############

        # Concatenación de las ramas del bloque de Inception para el cálculo de entropías
        concatenated_branches = concatenate(
            [branch_k1, branch_k2, branch_k3], axis=-1
        )
        concatenated_branches = tf.transpose(
            concatenated_branches, perm=(0, 3, 1, 2)
        )
        layer_entropy = Lambda(
            lambda x: renyi_entropy(x, alpha=alpha), name="entropy"
        )(concatenated_branches)

        layer_joint_entropy = Lambda(
            lambda x: joint_renyi_entropy(x, alpha=alpha),
            name="joint_entropy",
        )(concatenated_branches)

        concatenate_entropies = concatenate(
            [layer_entropy, layer_joint_entropy],
            axis=-1,
            name="concatenated_entropies",
        )

        ###############
        # Segunda capa convolucional con normalización por lotes y aplanamiento
        conv2D = Conv2D(F3, 3, padding='same', name='Conv2D_2')(inception)

        conv2D = BatchNormalization()(conv2D)
        flatten = Flatten(name='flatten')(conv2D)

        # Capas densas con activación y normalización
        dense = Dense(
            64, kernel_constraint=max_norm(norm_rate), activation="relu"
        )(flatten)
        dense = Dense(
            nb_classes, name='output', kernel_constraint=max_norm(norm_rate)
        )(dense)
        softmax = Activation('softmax', name='out_activation')(dense)

        # Creación del modelo
        model = Model(
            inputs=input1, outputs=[softmax, concatenate_entropies]
        )

        # Compilación del modelo
        model.compile(
            optimizer='adam',
            loss=[
                NormalizedBinaryCrossentropy(),
                RenyiMutualInformation(
                    C=tf.cast(64.0, tf.float64), name='MutualInfo'
                ),
            ],
            loss_weights=[0.8, 0.2],
            metrics=[['binary_accuracy'], [None]],
        )

        return model

    # ----------------------------------------------------------------------
    def __repr__(self):
        """"""
        return ""
