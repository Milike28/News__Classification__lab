# 🤖 Task 2: Transformer News Classification

Entrenamiento y comparación de modelos transformer (RoBERTa, DeBERTa, ModernBERT) en el dataset AG News.

---

## 📋 Descripción

Este proyecto implementa un sistema de clasificación de noticias usando tres modelos transformer state-of-the-art. El objetivo es comparar el rendimiento de diferentes arquitecturas en la tarea de clasificación multiclase de noticias.

### 🎯 Objetivos

1. Entrenar 3 modelos transformer en AG News dataset
2. Evaluar y comparar F1-scores
3. Analizar rendimiento por categoría
4. Generar visualizaciones comparativas
5. (Bonus) Clasificar noticias RPP con LLM

---

## 📊 Dataset: AG News

- **Fuente**: [Hugging Face - AG News](https://huggingface.co/datasets/ag_news)
- **Ejemplos**: 120,000 noticias
- **Categorías**: 4 clases
  - 0: World (Internacional)
  - 1: Sports (Deportes)
  - 2: Business (Negocios)
  - 3: Science/Tech (Ciencia/Tecnología)

### Split del Dataset

```
Train:      70% (84,000 ejemplos)
Validation: 15% (18,000 ejemplos)
Test:       15% (18,000 ejemplos)
```

---

## 🤖 Modelos Implementados

### 1. RoBERTa
- **Model ID**: `roberta-base`
- **Parámetros**: 125M
- **Arquitectura**: BERT optimizado
- **Ventaja**: Robusto, bien establecido

### 2. DeBERTa
- **Model ID**: `microsoft/deberta-v3-small`
- **Parámetros**: 86M
- **Arquitectura**: Disentangled attention
- **Ventaja**: Eficiente, buen ratio desempeño/tamaño

### 3. ModernBERT
- **Model ID**: `answerdotai/ModernBERT-base`
- **Parámetros**: 110M
- **Arquitectura**: BERT modernizado (2024)
- **Ventaja**: Arquitectura actualizada

---

## 🚀 Instalación y Uso

### Requisitos Previos

- Python 3.10+
- Google Colab (recomendado) o entorno local con GPU
- Google Drive (para almacenamiento persistente)

### 1️⃣ Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 2️⃣ Ejecutar en Google Colab

1. Subir `agnews_train_eval.ipynb` a Google Colab
2. Configurar GPU: `Runtime → Change runtime type → GPU (T4)`
3. Montar Google Drive
4. Ejecutar todas las celdas: `Runtime → Run all`

### 3️⃣ Estructura de Archivos

El notebook genera automáticamente:

```
H:\Mi unidad\News_Classification-lab\
├── models/
│   ├── roberta/best/      # Modelo RoBERTa entrenado
│   ├── deberta/best/      # Modelo DeBERTa entrenado
│   └── modernbert/best/   # Modelo ModernBERT entrenado
├── data/
│   ├── split_info.json    # Información del split
│   └── rpp_classified.json # (Bonus) Noticias RPP clasificadas
├── outputs/
│   ├── test_results.json  # Métricas de evaluación
│   ├── summary_table.csv  # Tabla resumen
│   └── analysis_report.md # Reporte de análisis
└── figures/
    ├── f1_comparison.png      # Comparación entre modelos
    ├── f1_per_class.png       # F1 por categoría
    └── f1_comparison_llm.png  # (Bonus) Comparación vs LLM
```

---

## 📈 Resultados

### F1-Scores en Test Set

| Modelo | F1 (Macro) | F1 (Weighted) | Tiempo Entrenamiento |
|--------|------------|---------------|---------------------|
| RoBERTa | 0.9495 | 0.9495 | 34 min |
| ModernBERT | 0.9463 | 0.9463 | 50 min |
| DeBERTa | 0.9444 | 0.9444 | 33 min |

### F1-Score por Categoría

```
World:        0.94
Sports:       0.96
Business:     0.93
Science/Tech: 0.95
```

---

## ⚡ Versión Rápida (10% Dataset)

Para desarrollo y pruebas rápidas, usa `agnews_train_eval_FAST.ipynb`:

- **Dataset**: 10% del original (12,000 ejemplos)
- **Tiempo**: 21 min (GPU) / 56 min (CPU)
- **F1 Score**: ~0.90-0.93 (ligeramente menor pero válido)

### Modificar Tamaño de Muestra

```python
# En la celda de split, cambiar:
sample_size = int(len(dataset['train']) * 0.10)  # 10%

# A:
sample_size = int(len(dataset['train']) * 0.25)  # 25% → 35 min
sample_size = int(len(dataset['train']) * 0.50)  # 50% → 60 min
sample_size = int(len(dataset['train']) * 1.00)  # 100% → 90 min
```

---

## 🎁 Bonus Task: Clasificación LLM

Clasificación de 50 noticias de RPP (Task 1) usando ChatGPT y comparación con modelos entrenados.

### Requisitos

- OpenAI API Key
- Noticias de RPP (de Task 1)
- Créditos en OpenAI (~$0.50)

### Configuración

```python
# Opción 1: Colab Secrets (recomendado)
# 🔑 → Add secret: OPENAI_API_KEY

# Opción 2: Input manual
from getpass import getpass
OPENAI_API_KEY = getpass('API Key: ')
```

### Resultados

- F1-Score: Comparación modelos vs LLM
- Análisis de divergencias
- Visualización comparativa

---

## 🛠️ Configuración Técnica

### Hiperparámetros

```python
{
    'max_length': 128,           # Tokens máximos
    'batch_size': 16,            # Batch size
    'learning_rate': 2e-5,       # Learning rate
    'epochs': 3,                 # Épocas
    'weight_decay': 0.01,        # Regularización
    'optimizer': 'adamw_torch'   # Optimizador
}
```

### Hardware Recomendado

- **GPU**: T4 (Google Colab gratis) o superior
- **RAM**: 12GB mínimo
- **Almacenamiento**: 2GB Drive para modelos

### Tiempos de Ejecución

**Con GPU T4:**
```
Setup:              3 min
Entrenamiento:     90 min (3 modelos × 30 min)
Evaluación:         5 min
Visualizaciones:    2 min
─────────────────────────
Total:            100 min
```

**Con CPU:**
```
Setup:              5 min
Entrenamiento:    300 min (3 modelos × 100 min)
Evaluación:        15 min
Visualizaciones:    2 min
─────────────────────────
Total:            322 min
```

---

## 📊 Métricas de Evaluación

### F1-Score (Macro)

```
F1 = 2 × (precision × recall) / (precision + recall)
F1_macro = promedio de F1 de todas las clases
```

**Usado para**: Comparación principal entre modelos

### F1-Score (Weighted)

```
F1_weighted = promedio ponderado por soporte de clase
```

**Usado para**: Análisis secundario considerando distribución

### Accuracy

```
Accuracy = predicciones correctas / total
```

**Usado para**: Bonus task (alineación con LLM)

---

## 🔧 Troubleshooting

### Error: "CUDA out of memory"

**Solución**:
```python
# Reducir batch size
'batch_size': 8  # en vez de 16
```

### Error: "fused=True requires..."

**Solución**:
```python
# Agregar en TrainingArguments:
optim="adamw_torch"
```

### Error: "Mountpoint must not already contain files"

**Solución**:
```python
# Verificar antes de montar
if os.path.exists('/content/drive/MyDrive'):
    print("✅ Drive ya montado")
else:
    drive.mount('/content/drive')
```

### Error: "No GPU available"

**Solución**:
```
Runtime → Change runtime type → GPU (T4)
```

### Entrenamiento muy lento

**Solución**:
```python
# Usar versión rápida (10% dataset)
# O reducir epochs:
'epochs': 2  # en vez de 3
```

---

## 📝 Archivos Principales

### Notebooks

- `agnews_train_eval.ipynb` - Versión completa (100% dataset)
- `agnews_train_eval_FAST.ipynb` - Versión rápida (10% dataset)

### Scripts

- `train_model()` - Función de entrenamiento unificada
- `compute_metrics()` - Cálculo de F1-scores
- `classify_with_llm()` - Clasificación con ChatGPT

### Outputs

- `test_results.json` - Métricas detalladas
- `summary_table.csv` - Tabla resumen
- `analysis_report.md` - Análisis completo
- `f1_comparison.png` - Gráfica comparativa
- `f1_per_class.png` - Gráfica por categoría

---

## 🎓 Requisitos de Entrega

### ✅ Obligatorios

- [x] Dataset AG News split 70/15/15
- [x] 3 modelos transformer entrenados
- [x] F1-scores calculados (macro y weighted)
- [x] Test set usado solo UNA vez
- [x] Gráfica de comparación
- [x] Tabla resumen
- [x] Reporte de análisis
- [x] Código reproducible

### ⭐ Bonus (+3 pts)

- [x] Clasificación LLM de noticias RPP
- [x] Comparación modelos vs LLM
- [x] Análisis de divergencias
- [x] Visualización comparativa

---

## 📚 Referencias

### Datasets

- [AG News Dataset](https://huggingface.co/datasets/ag_news)

### Modelos

- [RoBERTa Paper](https://arxiv.org/abs/1907.11692)
- [DeBERTa Paper](https://arxiv.org/abs/2006.03654)
- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)

### Librerías

- [Transformers](https://huggingface.co/docs/transformers)
- [Datasets](https://huggingface.co/docs/datasets)
- [PyTorch](https://pytorch.org/)

---

## 👥 Autores

Proyecto desarrollado como parte del curso de Data Science.

---

## 📄 Licencia

Este proyecto es con fines educativos.

---

## 🆘 Soporte

Para problemas o preguntas:

1. Revisar sección Troubleshooting
2. Verificar que GPU está habilitada
3. Confirmar que Drive está montado
4. Revisar logs de error completos

---

## 📊 Rúbrica

| Criterio | Puntos | Status |
|----------|--------|--------|
| Data & Reproducibility | 4 pts | ✅ |
| Task 2: Transformer Models | 6 pts | ✅ |
| Visualization & Comparison | 2 pts | ✅ |
| Bonus: LLM Classification | +3 pts | ⭐ |
| **Total** | **15 pts** | ✅ |

---

**Desarrollado**: Noviembre 2025  
**Última actualización**: 15 de Noviembre, 2025  
**Versión**: 1.0  
**Status**: ✅ Producción