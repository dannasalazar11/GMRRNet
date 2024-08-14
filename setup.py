from setuptools import setup, find_packages

setup(
    name="gmrrnet",  # Nombre del paquete
    version="0.1.0",  # Versión del paquete
    packages=find_packages(),  # Busca automáticamente los paquetes en el directorio
    install_requires=[
        'matplotlib',
        'mne',
        'scikit-learn',
        'scipy',
        'tensorflow',
        'keras-tuner',
        'numpy',
        'pandas',
        'IPython'
    ],
    author="Tu nombre",
    author_email="tu_email@example.com",
    description="Implementación del modelo GMRRNet",
    long_description=open("README.md").read(),  # Lee el contenido de README.md para la descripción
    long_description_content_type="text/markdown",  # El contenido es de tipo Markdown
    url="https://github.com/dannasalazar11/GMRRNet",  # URL del repositorio
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',  # Requisito mínimo de Python
)
