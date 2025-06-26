# Estructura del Repositorio: Simulador Predictivo de Impacto Financiero

```
simulador-impacto-financiero/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── config/
│   ├── __init__.py
│   ├── settings.py
│   └── financial_parameters.json
├── src/
│   ├── __init__.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── financial_model.py
│   │   ├── scenario_generator.py
│   │   └── risk_calculator.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py
│   │   └── data_validator.py
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── charts.py
│   │   ├── dashboard.py
│   │   └── reports.py
│   └── utils/
│       ├── __init__.py
│       ├── helpers.py
│       └── constants.py
├── data/
│   ├── raw/
│   │   └── sample_data.csv
│   ├── processed/
│   └── examples/
│       ├── scenario_basic.json
│       ├── scenario_expansion.json
│       └── scenario_crisis.json
├── notebooks/
│   ├── 01_exploratory_analysis.ipynb
│   ├── 02_model_development.ipynb
│   ├── 03_scenario_testing.ipynb
│   └── 04_results_visualization.ipynb
├── tests/
│   ├── __init__.py
│   ├── test_financial_model.py
│   ├── test_scenario_generator.py
│   └── test_data_validation.py
├── docs/
│   ├── methodology.md
│   ├── user_guide.md
│   ├── api_reference.md
│   └── deployment.md
├── web/
│   ├── index.html
│   ├── css/
│   │   └── styles.css
│   ├── js/
│   │   └── simulator.js
│   └── assets/
│       └── images/
└── examples/
    ├── basic_simulation.py
    ├── advanced_scenarios.py
    └── custom_parameters.py
```

## Descripción de Directorios

### Directorio Raíz
- **README.md**: Documentación principal del proyecto
- **requirements.txt**: Dependencias de Python
- **LICENSE**: Licencia del proyecto
- **.gitignore**: Archivos a ignorar en Git

### config/
Configuraciones del sistema y parámetros financieros

### src/
Código fuente principal organizado por funcionalidad:
- **core/**: Lógica central del simulador
- **data/**: Manejo y validación de datos
- **visualization/**: Generación de gráficos y reportes
- **utils/**: Utilidades y constantes

### data/
Datos del proyecto organizados por estado de procesamiento

### notebooks/
Jupyter notebooks para análisis y desarrollo

### tests/
Pruebas unitarias del código

### docs/
Documentación técnica y metodológica

### web/
Interface web para el simulador

### examples/
Ejemplos de uso y casos prácticos