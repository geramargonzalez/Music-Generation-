# 🎵 Generación de Música con Deep Learning (RNN/LSTM)

**Proyecto académico — Curso de Deep Learning | UTEC × MIT**  
Especialización en Ciencia de Datos y Machine Learning

**Autor:** Gerardo González

---

## Descripción

Este proyecto implementa un modelo de **Red Neuronal Recurrente (RNN)** basado en la arquitectura **LSTM (Long Short-Term Memory)** para la **generación automática de música**. El modelo aprende patrones a partir de partituras codificadas en [notación ABC](https://en.wikipedia.org/wiki/ABC_notation) y, una vez entrenado, es capaz de componer nuevas melodías de forma autónoma.

El notebook está basado en el **Lab 1 – Part 2** del curso [MIT Introduction to Deep Learning (6.S191)](http://introtodeeplearning.com), adaptado y documentado como proyecto de la especialización en Ciencia de Datos y Machine Learning de la **UTEC**.

---

## Estructura del Proyecto

```
DeepMusic Generation/
├── Gerardo_Gonzalez_Music_Generation.ipynb   # Notebook principal del proyecto
├── Gerardo_Gonzalez_Music_Generation_vc.ipynb # Versión de control del notebook
├── deepMusicGeneration.ipynb                  # Notebook exploratorio
├── PT_Part1_Intro.ipynb                       # Parte 1: Introducción a PyTorch
├── training_checkpoints/                      # Checkpoints del modelo entrenado
├── output_*.wav                               # Archivos de audio generados
└── README.md                                  # Este archivo
```

---

## Pipeline del Proyecto

### 1. Configuración del Entorno
- Instalación de dependencias: **PyTorch**, **Comet ML** (tracking de experimentos), **mitdeeplearning** (utilidades del laboratorio MIT).
- Herramientas de sistema para síntesis de audio: `abcmidi` (ABC → MIDI) y `timidity` (MIDI → WAV).

### 2. Carga y Exploración del Dataset
- Se carga un corpus de canciones en **notación ABC** mediante `mitdeeplearning`.
- Se puede escuchar cualquier canción del dataset convirtiéndola a audio.

### 3. Preprocesamiento
- **Tokenización a nivel de caracter:** se construye un vocabulario con todos los caracteres únicos del corpus (~83 caracteres).
- **Mapeos** `char2idx` / `idx2char` para convertir entre texto y representación numérica.
- **Vectorización** del texto completo del dataset.
- **Generación de batches:** pares (input, target) donde el target es la secuencia desplazada un caracter a la derecha (predicción del siguiente caracter).

### 4. Arquitectura del Modelo (LSTM)

| Capa | Descripción | Dimensión |
|------|-------------|-----------|
| `nn.Embedding` | Convierte IDs de caracteres en vectores densos aprendidos | `vocab_size → 256` |
| `nn.LSTM` | Procesa la secuencia manteniendo memoria temporal | `256 → 1024` |
| `nn.Linear` | Proyecta los estados ocultos a logits sobre el vocabulario | `1024 → vocab_size` |

**Salida del modelo:** `(batch_size, seq_length, vocab_size)` — logits para cada posición temporal.

### 5. Entrenamiento

- **Función de pérdida:** `CrossEntropyLoss` (clasificación multi-clase caracter por caracter).
- **Optimizador:** Adam con `learning_rate = 2e-3`.
- **Hiperparámetros principales:**

| Parámetro | Valor |
|-----------|-------|
| Iteraciones de entrenamiento | 6,000 |
| Tamaño de batch | 32 |
| Longitud de secuencia | 200 |
| Learning rate | 2e-3 |
| Dimensión de embedding | 256 |
| Tamaño oculto LSTM | 1024 |

- **Tracking:** métricas de pérdida registradas en [Comet ML](https://www.comet.com/) para monitoreo en tiempo real.
- **Checkpoints:** guardados cada 100 iteraciones.

### 6. Generación de Música

Se implementaron dos técnicas de muestreo para la generación de texto:

- **Muestreo multinomial estándar:** basado en softmax sobre los logits.
- **Muestreo Nucleus (Top-p) con temperatura:** filtra los caracteres menos probables reteniendo solo el núcleo de probabilidad acumulada hasta `p`, y controla la aleatoriedad con un parámetro de temperatura.
  - `temperature < 1.0` → generación más conservadora y válida.
  - `temperature > 1.0` → generación más creativa pero con mayor riesgo de errores sintácticos.

El texto generado en notación ABC se convierte a audio (MIDI → WAV) para su reproducción.

### 7. Resultados

El modelo entrenado genera múltiples canciones con estructura ABC válida, incluyendo:
- Headers correctos (`X:`, `T:`, `M:`, `L:`, `K:`)
- Barras de compás (`|`, `:|`)
- Patrones melódicos y rítmicos coherentes

**Canciones generadas de ejemplo:**

| Canción | Duración | Tempo estimado | Timbre (centroide espectral) |
|---------|----------|----------------|------------------------------|
| Song 0 | ~28.1 s | ~120 BPM (rápido) | ~3397 Hz (brillante) |
| Song 1 | ~67.7 s | ~60 BPM (lento) | ~2947 Hz (cálido/oscuro) |

---

## Tecnologías Utilizadas

- **Python 3**
- **PyTorch** — framework de deep learning
- **Comet ML** — tracking de experimentos
- **NumPy** — manipulación numérica
- **SciPy** — escritura de archivos WAV
- **tqdm** — barras de progreso
- **mitdeeplearning** — utilidades del curso MIT 6.S191
- **abcmidi / timidity** — conversión ABC → MIDI → Audio

---

## Cómo Ejecutar

1. **Abrir el notebook** `Gerardo_Gonzalez_Music_Generation.ipynb` en Google Colab o en un entorno local con GPU.
2. **Configurar la API key de Comet ML** (registrarse en [comet.com](https://www.comet.com/) y obtener una clave personal).
3. **Ejecutar todas las celdas** en orden secuencial.
4. Los archivos de audio generados (`output_*.wav`) se guardarán localmente y se registrarán en Comet ML.

> **Nota:** Se recomienda usar un entorno con GPU (Google Colab con runtime GPU) para acelerar el entrenamiento.

---

## Contexto Académico

| | |
|---|---|
| **Institución** | UTEC (Universidad de Ingeniería y Tecnología) |
| **Curso** | Deep Learning (MIT Introduction to Deep Learning — 6.S191) |
| **Especialización** | Ciencia de Datos y Machine Learning |
| **Laboratorio** | Lab 1, Parte 2 — Generación de Música con RNNs |
| **Autor** | Gerardo González |

---

## Referencias

- [MIT Introduction to Deep Learning (6.S191)](http://introtodeeplearning.com)
- [Repositorio del curso en GitHub](https://github.com/MITDeepLearning/introtodeeplearning)
- [Documentación de PyTorch](https://pytorch.org/docs/stable/)
- [Notación ABC — Wikipedia](https://en.wikipedia.org/wiki/ABC_notation)
- [Comet ML — Documentación](https://www.comet.com/docs/v2/)

---

## Licencia

© MIT Introduction to Deep Learning — [http://introtodeeplearning.com](http://introtodeeplearning.com)
