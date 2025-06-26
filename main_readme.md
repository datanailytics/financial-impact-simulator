# Simulador Predictivo de Impacto Financiero de Decisiones Estratégicas

## Descripción del Proyecto

El Simulador Predictivo de Impacto Financiero es una herramienta analítica avanzada diseñada para evaluar el impacto económico de decisiones estratégicas empresariales. Utiliza modelos financieros robustos y análisis de escenarios para proporcionar proyecciones precisas que apoyan la toma de decisiones informada.

## Características Principales

### Modelado Financiero Avanzado
- Proyecciones de flujo de caja con múltiples horizontes temporales
- Análisis de sensibilidad y escenarios Monte Carlo
- Cálculo automatizado de métricas financieras clave (VPN, TIR, Payback)

### Generación de Escenarios
- Simulación de escenarios optimistas, pesimistas y realistas
- Incorporación de variables macroeconómicas y de mercado
- Análisis de riesgo y incertidumbre

### Visualización Interactiva
- Dashboards dinámicos con gráficos interactivos
- Reportes automatizados en formato PDF y Excel
- Interface web responsiva para análisis en tiempo real

### Flexibilidad y Personalización
- Parámetros configurables según industria y contexto
- Integración con fuentes de datos externas
- API para integración con sistemas empresariales

## Instalación y Configuración

### Requisitos del Sistema
- Python 3.8 o superior
- Memoria RAM mínima: 4GB
- Espacio en disco: 2GB

### Instalación

1. **Clona el repositorio**
```bash
git clone https://github.com/tu-usuario/simulador-impacto-financiero.git
cd simulador-impacto-financiero
```

2. **Crea un entorno virtual**
```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

3. **Instala las dependencias**
```bash
pip install -r requirements.txt
```

4. **Configura los parámetros**
```bash
cp config/financial_parameters.json.example config/financial_parameters.json
# Edita el archivo según tus necesidades
```

### Configuración Inicial

La configuración del simulador se realiza mediante el archivo `config/financial_parameters.json`:

```json
{
  "simulation_parameters": {
    "time_horizon": 5,
    "discount_rate": 0.10,
    "tax_rate": 0.25,
    "inflation_rate": 0.03
  },
  "risk_parameters": {
    "market_volatility": 0.15,
    "operational_risk": 0.10,
    "financial_risk": 0.08
  }
}
```

## Uso del Simulador

### Ejemplo Básico

```python
from src.core.financial_model import FinancialModel
from src.core.scenario_generator import ScenarioGenerator

# Inicializar el modelo
model = FinancialModel()

# Cargar datos de entrada
model.load_base_scenario("data/examples/scenario_basic.json")

# Ejecutar simulación
results = model.run_simulation()

# Generar reportes
model.generate_report(results, output_format="pdf")
```

### Análisis de Escenarios Múltiples

```python
from src.core.scenario_generator import ScenarioGenerator

# Generar múltiples escenarios
generator = ScenarioGenerator()
scenarios = generator.generate_scenarios(
    base_case="data/examples/scenario_basic.json",
    variations=["optimistic", "pessimistic", "crisis"]
)

# Ejecutar simulaciones comparativas
comparative_results = model.run_comparative_analysis(scenarios)
```

### Interface Web

Para utilizar la interface web interactiva:

```bash
python -m http.server 8000 --directory web
```

Accede a `http://localhost:8000` para utilizar el simulador desde el navegador.

## Estructura del Proyecto

La arquitectura del proyecto sigue principios de desarrollo modular y separación de responsabilidades:

- **src/core/**: Contiene la lógica central del simulador financiero
- **src/data/**: Módulos para carga y validación de datos
- **src/visualization/**: Generación de gráficos y reportes
- **notebooks/**: Análisis exploratorio y desarrollo de modelos
- **web/**: Interface web responsiva
- **tests/**: Suite completa de pruebas unitarias

## Metodología

### Modelado Financiero
El simulador implementa modelos de valoración estándar de la industria financiera, incluyendo:

- **Descuento de Flujos de Caja (DCF)**: Para valoración de proyectos de inversión
- **Análisis de Sensibilidad**: Evaluación del impacto de cambios en variables clave
- **Simulación Monte Carlo**: Análisis probabilístico de resultados

### Gestión de Riesgo
La evaluación de riesgo incorpora:

- **Riesgo de Mercado**: Volatilidad de precios y demanda
- **Riesgo Operacional**: Variabilidad en costos y eficiencia operativa
- **Riesgo Financiero**: Impacto de estructura de capital y financiamiento

### Validación de Modelos
Todos los modelos incluyen:

- Validación cruzada con datos históricos
- Pruebas de estrés en escenarios extremos
- Calibración periódica de parámetros

## Ejemplos de Uso

### Evaluación de Expansión de Negocio
```python
# Cargar escenario de expansión
expansion_scenario = "data/examples/scenario_expansion.json"
results = model.evaluate_expansion(expansion_scenario)

# Analizar métricas clave
print(f"VPN del proyecto: ${results.npv:,.2f}")
print(f"TIR: {results.irr:.2%}")
print(f"Período de recuperación: {results.payback:.1f} años")
```

### Análisis de Impacto de Crisis
```python
# Simular impacto de crisis económica
crisis_impact = model.stress_test(
    scenario="economic_downturn",
    severity=0.3,
    duration=2
)

# Evaluar medidas de mitigación
mitigation_strategies = model.evaluate_mitigations(crisis_impact)
```

## Contribuciones

Las contribuciones son bienvenidas. Por favor, sigue estos pasos:

1. Fork el repositorio
2. Crea una rama para tu característica (`git checkout -b feature/nueva-caracteristica`)
3. Commit tus cambios (`git commit -am 'Agrega nueva característica'`)
4. Push a la rama (`git push origin feature/nueva-caracteristica`)
5. Crea un Pull Request

### Estándares de Código
- Sigue PEP 8 para estilo de código Python
- Incluye docstrings para todas las funciones y clases
- Añade pruebas unitarias para nuevas funcionalidades
- Actualiza la documentación según corresponda

## Licencia

Este proyecto está licenciado bajo la Licencia MIT. Ver el archivo `LICENSE` para más detalles.

## Contacto

Para preguntas o sugerencias, puedes contactar a través de:
- Issues de GitHub: [Reportar un problema](https://github.com/tu-usuario/simulador-impacto-financiero/issues)
- Email: tu-email@ejemplo.com

## Versión

**Versión actual**: 1.0.0

### Historial de Versiones
- **1.0.0** (2025-05-31): Lanzamiento inicial
  - Simulador básico de impacto financiero
  - Interface web interactiva
  - Documentación completa
  - Suite de pruebas unitarias