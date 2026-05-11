# BreastAI — Clasificación de Cáncer de Mama

**Proyecto Final Individual** · Entrega: 11-Mayo-2026

## 🚀 Demo en Vivo
Despliega en GitHub Pages → `https://tuusuario.github.io/breast-cancer-ai/`

---

## 📋 Descripción
Página web de clasificación de cáncer de mama que utiliza dos modelos de ML entrenados con el **Breast Cancer Wisconsin Dataset** de UCI:

| Modelo | Accuracy |
|--------|----------|
| Regresión Logística | 97.37% |
| Red Neuronal (MLP 64→32) | 97.37% |

---

## ⚙️ Características
- **Predicción Individual**: Ingresa los 30 biomarcadores → obtiene diagnóstico con confianza
- **Predicción por Lotes**: Sube CSV o usa datos de prueba → matriz de confusión + métricas
- Los modelos corren 100% en el **navegador** (sin servidor) usando pesos exportados a JSON

---

## 🛠 Dataset
- **Fuente**: [UCI ML Repository - Breast Cancer Wisconsin](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- **Muestras**: 569 (80% train / 20% test)
- **Features**: 30 características del núcleo celular
- **Clases**: Maligno (0) / Benigno (1)

---

## 📁 Estructura
```
/
├── index.html          # Página principal
├── css/
│   └── style.css       # Estilos (tema fansappy dark)
├── js/
│   ├── app.js          # Lógica de inferencia y UI
│   └── model_data.json # Pesos exportados (LR + MLP)
├── train_models.py     # Script de entrenamiento (Python)
└── README.md
```

---

## 🌐 Despliegue en GitHub Pages

1. Crea un repositorio en GitHub
2. Sube todos los archivos
3. Ve a **Settings → Pages → Source: main branch**
4. ¡Listo! Tu app estará en `https://tuusuario.github.io/nombre-repo/`

---

## 🤖 Entrenamiento local
```bash
pip install scikit-learn numpy
python train_models.py
```

---

## ⚠️ Aviso
Esta herramienta es **únicamente para fines educativos**. No sustituye diagnóstico médico profesional.
