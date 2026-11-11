# MLOps end-to-end: Predicción de renovación de préstamos

**Caso académico:** estimar `P(FLAG_VENTA=1)` para priorizar acciones de renovación. No se utiliza para decisiones automáticas de otorgamiento crediticio.

## Arquitectura

`CSV original → Validación de esquema → Split temporal → Preprocesamiento → Entrenamiento → Validación/umbral → Test final → MLflow → Artefacto versionable → API FastAPI → Prometheus → Monitoreo básico → GitHub Actions`

## Datos

El dataset de clase contiene **87,556 filas y 22 columnas**, meses enero-septiembre de 2015. `FLAG_VENTA` tiene 3,492 positivos (~3.99%). **Por confidencialidad, el CSV original no se incluye en Git.** Copia el archivo facilitado en clase a `data/raw/Dataset Renovacion_prestamo.csv`. Se lee con separador `;` y codificación `latin1`.

Excluimos `CLIENTE` (identificador), `MES` (tiempo de partición) y `FLAG_VENTA` (objetivo) del vector de entrada. Tratamiento de faltantes numéricos con mediana e indicador; categóricos con moda y OneHotEncoder. Regresión logística balanceada como modelo reproducible e interpretable.

## Inicio rápido

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python -m src.train --mlflow
pytest -q
uvicorn src.api:app --reload
```

Visita `http://localhost:8000/docs`, `/health`, `/ready` y `/metrics`. Para ver experimentos: `mlflow ui`.

## Estrategia de evaluación

- Entrenamiento: enero-julio 2015; validación: agosto 2015; prueba: septiembre 2015.
- Selección de umbral según F1 únicamente en validación.
- Métricas: average precision (PR-AUC), ROC-AUC, precision, recall, F1 y matriz de confusión.
- **El test no se utiliza para elegir parámetros ni umbral.**
- Nota: si los mismos clientes aparecen en varios meses, la evaluación representa capacidad temporal para clientes recurrentes y no independencia por cliente nuevo.

## Ejemplo de predicción

Los campos de la petición deben coincidir con las 19 variables del modelo y los valores deben corresponder a una solicitud. El esquema dinámico evita valores de demostración arbitrarios. Puedes extraer una fila del CSV sin `MES`, `CLIENTE`, `FLAG_VENTA`:

```python
import pandas as pd,requests
row=pd.read_csv('data/raw/Dataset Renovacion_prestamo.csv',sep=';',encoding='latin1').iloc[0]
features=row.drop(labels=['MES','CLIENTE','FLAG_VENTA']).where(lambda x:x.notna(),None).to_dict()
features={k:(None if pd.isna(v) else v.item() if hasattr(v,'item') else v) for k,v in features.items()}
print(requests.post('http://localhost:8000/predict',json={'features':features}).json())
```

## Monitoreo y operación

`/metrics` expone peticiones y latencia para Prometheus; `monitoring/predictions.jsonl` almacena solo el score y marca predicha, sin identificadores personales. `python -m src.monitor --data <ruta CSV reciente>` alerta ante cambios de >10 puntos porcentuales en la tasa de faltantes por variable. Un monitor de **drift estadístico de distribuciones**, resultados reales, alertas, retraining periódico y CD de producción son extensiones futuras, **no se presentan como implementadas**.

## CI, Docker y despliegue

GitHub Actions ejecuta pruebas, compilación y build Docker en cada push/PR. No existe despliegue automatizado a nube. Para ejecución local: `docker compose up --build` tras entrenar y disponer del artefacto `artifacts/model.joblib`. **La carpeta artifacts está excluida de Git**; en un entorno nuevo, entrenar antes de construir la imagen o descargar un artefacto autorizado. No subas datos bancarios ni secretos.

## Riesgos y gobierno

Los datos son históricos y el modelo debe recalibrarse antes de cualquier uso real. `SEXO`, `EDAD` y `REGION` requieren evaluación de equidad, fundamentos legales y controles de privacidad. El score es apoyo comercial académico, no decisión financiera automática.

## Estructura

```text
src/common.py         Datos, validación y partición
src/train.py          Pipeline, métricas y tracking MLflow
src/api.py            Servicio REST, health y métricas
src/monitor.py        Control de cambios en faltantes
tests/               Pruebas automatizadas
.github/workflows/ci.yml  Integración continua
Dockerfile, docker-compose.yml  Contenerización
artifacts/            Modelo y métricas (solo local)
data/raw/             Datos originales (solo local)
```
