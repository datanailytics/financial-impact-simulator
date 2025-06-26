#!/usr/bin/env python3
"""
Ejemplo básico de uso del Simulador Predictivo de Impacto Financiero.

Este script demuestra cómo configurar y ejecutar una simulación financiera
para evaluar el impacto de una decisión estratégica de expansión de negocio.

Escenario de ejemplo: Evaluación de apertura de nueva línea de productos
- Inversión inicial: $500,000
- Horizonte temporal: 5 años
- Proyecciones de ingresos y costos definidas

Autor: [Tu Nombre]
Fecha: Mayo 2025
"""

import sys
import os
import json
from pathlib import Path

# Agregar el directorio src al path para importar módulos
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from core.financial_model import FinancialModel, FinancialResults
from visualization.charts import create_financial_dashboard
from utils.helpers import format_currency, validate_scenario_data


def create_example_scenario() -> dict:
    """
    Crea un escenario de ejemplo para la simulación.
    
    Representa la evaluación de lanzamiento de una nueva línea de productos
    con proyecciones conservadoras basadas en análisis de mercado.
    
    Returns:
        dict: Diccionario con datos del escenario de simulación
    """
    scenario = {
        "project_name": "Expansión Nueva Línea de Productos",
        "project_description": "Lanzamiento de línea premium en mercado objetivo",
        "analyst": "Equipo de Análisis Financiero",
        "creation_date": "2025-05-31",
        
        # Parámetros principales de inversión
        "initial_investment": 500000,  # Inversión inicial en USD
        "time_horizon": 5,  # Años de proyección
        
        # Proyecciones de ingresos anuales (USD)
        "revenue_projections": [
            150000,  # Año 1: Penetración inicial del mercado
            280000,  # Año 2: Crecimiento acelerado
            420000,  # Año 3: Consolidación en el mercado
            550000,  # Año 4: Expansión a segmentos adicionales
            650000   # Año 5: Madurez del producto
        ],
        
        # Proyecciones de costos operativos anuales (USD)
        "cost_projections": [
            90000,   # Año 1: Costos de lanzamiento elevados
            165000,  # Año 2: Economías de escala iniciales
            240000,  # Año 3: Costos estabilizados
            300000,  # Año 4: Incremento por expansión
            350000   # Año 5: Optimización de procesos
        ],
        
        # Parámetros adicionales del proyecto
        "market_assumptions": {
            "target_market_size": 10000000,  # Tamaño del mercado objetivo
            "expected_market_share": 0.065,  # Participación esperada (6.5%)
            "competitive_intensity": "medium",
            "regulatory_environment": "stable"
        },
        
        # Factores de riesgo específicos del proyecto
        "project_risks": {
            "market_acceptance_risk": 0.15,
            "competition_risk": 0.12,
            "technology_risk": 0.08,
            "regulatory_risk": 0.05
        },
        
        # Supuestos operativos
        "operational_assumptions": {
            "staff_required": 8,
            "facility_requirements": "expansion_existing",
            "technology_investments": 75000,
            "marketing_budget_annual": 25000
        }
    }
    
    return scenario


def save_scenario_to_file(scenario: dict, filename: str = "data/examples/scenario_basic.json") -> None:
    """
    Guarda el escenario en un archivo JSON para reutilización futura.
    
    Args:
        scenario (dict): Datos del escenario a guardar
        filename (str): Ruta del archivo donde guardar el escenario
    """
    # Crear directorio si no existe
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(scenario, f, indent=2, ensure_ascii=False)
    
    print(f"Escenario guardado en: {filename}")


def run_basic_simulation():
    """
    Ejecuta una simulación básica completa y muestra los resultados.
    
    Esta función demuestra el flujo completo desde la configuración
    hasta la generación de reportes y visualizaciones.
    """
    print("=" * 60)
    print("SIMULADOR PREDICTIVO DE IMPACTO FINANCIERO")
    print("Ejemplo de Simulación Básica")
    print("=" * 60)
    print()
    
    try:
        # Paso 1: Inicializar el modelo financiero
        print("1. Inicializando modelo financiero...")
        model = FinancialModel(config_path="config/financial_parameters.json")
        print("   ✓ Modelo inicializado correctamente")
        print()
        
        # Paso 2: Crear y validar escenario de ejemplo
        print("2. Creando escenario de simulación...")
        scenario = create_example_scenario()
        
        # Validar datos del escenario
        if validate_scenario_data(scenario):
            print("   ✓ Escenario validado correctamente")
        else:
            print("   ✗ Error en validación del escenario")
            return
        
        # Guardar escenario para futuras referencias
        save_scenario_to_file(scenario)
        print()
        
        # Paso 3: Ejecutar simulación
        print("3. Ejecutando simulación financiera...")
        results = model.run_simulation(scenario)
        print("   ✓ Simulación completada exitosamente")
        print()
        
        # Paso 4: Mostrar resultados principales
        print("4. RESULTADOS DE LA SIMULACIÓN")
        print("-" * 30)
        display_main_results(results)
        print()
        
        # Paso 5: Análisis de sensibilidad
        print("5. ANÁLISIS DE SENSIBILIDAD")
        print("-" * 30)
        display_sensitivity_analysis(results.sensitivity_analysis)
        print()
        
        # Paso 6: Evaluación de riesgo
        print("6. EVALUACIÓN DE RIESGO")
        print("-" * 30)
        display_risk_assessment(results.risk_metrics)
        print()
        
        # Paso 7: Recomendaciones
        print("7. RECOMENDACIONES")
        print("-" * 30)
        generate_recommendations(results, scenario)
        print()
        
        # Paso 8: Generar reporte completo
        print("8. Generando reporte completo...")
        report = model.generate_report(results, output_format="summary")
        
        # Guardar reporte en archivo
        report_filename = f"reports/simulation_report_{scenario['creation_date']}.txt"
        Path(report_filename).parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"   ✓ Reporte guardado en: {report_filename}")
        print()
        
        print("=" * 60)
        print("SIMULACIÓN COMPLETADA EXITOSAMENTE")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error durante la simulación: {str(e)}")
        raise


def display_main_results(results: FinancialResults) -> None:
    """
    Muestra los resultados principales de la simulación de forma estructurada.
    
    Args:
        results (FinancialResults): Resultados de la simulación financiera
    """
    print(f"Valor Presente Neto (VPN): {format_currency(results.npv)}")
    
    if not pd.isna(results.irr):
        print(f"Tasa Interna de Retorno (TIR): {results.irr:.2%}")
    else:
        print("Tasa Interna de Retorno (TIR): No calculable")
    
    if results.payback_period != float('inf'):
        print(f"Período de Recuperación: {results.payback_period:.1f} años")
    else:
        print("Período de Recuperación: No se recupera la inversión")
    
    print(f"Índice de Rentabilidad: {results.profitability_index:.2f}")
    
    # Mostrar evaluación cualitativa
    if results.npv > 0:
        viability = "VIABLE"
        symbol = "✓"
    else:
        viability = "NO VIABLE"
        symbol = "✗"
    
    print(f"\nEvaluación General: {symbol} Proyecto {viability}")


def display_sensitivity_analysis(sensitivity_data: dict) -> None:
    """
    Presenta los resultados del análisis de sensibilidad.
    
    Args:
        sensitivity_data (dict): Datos del análisis de sensibilidad
    """
    print("Variables críticas identificadas:")
    
    for variable, scenarios in sensitivity_data.items():
        print(f"\n{variable.replace('_', ' ').title()}:")
        
        # Encontrar el rango de variación del VPN
        vpn_values = list(scenarios.values())
        vpn_min = min(vpn_values)
        vpn_max = max(vpn_values)
        sensitivity_range = vpn_max - vpn_min
        
        print(f"  Rango de VPN: {format_currency(vpn_min)} a {format_currency(vpn_max)}")
        print(f"  Sensibilidad: {format_currency(sensitivity_range)}")
        
        # Determinar nivel de criticidad
        if sensitivity_range > 100000:
            criticality = "ALTA"
        elif sensitivity_range > 50000:
            criticality = "MEDIA"
        else:
            criticality = "BAJA"
        
        print(f"  Criticidad: {criticality}")


def display_risk_assessment(risk_metrics: dict) -> None:
    """
    Presenta la evaluación de riesgo del proyecto.
    
    Args:
        risk_metrics (dict): Métricas de riesgo calculadas
    """
    overall_risk = risk_metrics.get('overall_risk_score', 0)
    
    # Clasificar nivel de riesgo
    if overall_risk < 0.3:
        risk_level = "BAJO"
        risk_color = "verde"
    elif overall_risk < 0.6:
        risk_level = "MEDIO"
        risk_color = "amarillo"
    else:
        risk_level = "ALTO"
        risk_color = "rojo"
    
    print(f"Nivel de Riesgo General: {risk_level} ({overall_risk:.1%})")
    print(f"Semáforo de Riesgo: {risk_color.upper()}")
    
    print("\nFactores de Riesgo Detallados:")
    print(f"  Volatilidad de Flujos de Caja: {risk_metrics.get('cash_flow_volatility', 0):.1%}")
    print(f"  Riesgo de Mercado: {risk_metrics.get('market_risk_factor', 0):.1%}")
    print(f"  Riesgo Operacional: {risk_metrics.get('operational_risk_factor', 0):.1%}")
    print(f"  Riesgo Financiero: {risk_metrics.get('financial_risk_factor', 0):.1%}")


def generate_recommendations(results: FinancialResults, scenario: dict) -> None:
    """
    Genera recomendaciones basadas en los resultados de la simulación.
    
    Args:
        results (FinancialResults): Resultados de la simulación
        scenario (dict): Datos del escenario evaluado
    """
    recommendations = []
    
    # Análisis de viabilidad financiera
    if results.npv > 0:
        recommendations.append("El proyecto es financieramente viable según las métricas de VPN.")
    else:
        recommendations.append("El proyecto no es viable financieramente. Considerar modificaciones o alternativas.")
    
    # Análisis de rentabilidad
    if results.profitability_index > 1.2:
        recommendations.append("El proyecto muestra excelente rentabilidad. Proceder con implementación.")
    elif results.profitability_index > 1.0:
        recommendations.append("El proyecto es rentable pero marginal. Evaluar optimizaciones.")
    else:
        recommendations.append("La rentabilidad es insuficiente. Revisar supuestos y estrategia.")
    
    # Análisis de período de recuperación
    if results.payback_period <= 3:
        recommendations.append("El período de recuperación es atractivo para inversionistas.")
    elif results.payback_period <= 5:
        recommendations.append("El período de recuperación es aceptable para este tipo de proyecto.")
    else:
        recommendations.append("El período de recuperación es extenso. Considerar alternativas.")
    
    # Análisis de riesgo
    risk_score = results.risk_metrics.get('overall_risk_score', 0)
    if risk_score > 0.6:
        recommendations.append("El nivel de riesgo es elevado. Implementar estrategias de mitigación.")
    elif risk_score > 0.3:
        recommendations.append("El riesgo es moderado. Monitorear factores críticos.")
    else:
        recommendations.append("El perfil de riesgo es favorable para la inversión.")
    
    # Mostrar recomendaciones
    for i, recommendation in enumerate(recommendations, 1):
        print(f"{i}. {recommendation}")


def run_comparative_scenarios():
    """
    Ejecuta una comparación entre múltiples escenarios para análisis avanzado.
    
    Esta función demuestra cómo evaluar diferentes alternativas estratégicas
    utilizando variaciones del escenario base.
    """
    print("\n" + "=" * 60)
    print("ANÁLISIS COMPARATIVO DE ESCENARIOS")
    print("=" * 60)
    
    model = FinancialModel()
    base_scenario = create_example_scenario()
    
    # Crear variaciones del escenario
    scenarios = {
        "Conservador": modify_scenario_for_comparison(base_scenario, "conservative"),
        "Base": base_scenario,
        "Optimista": modify_scenario_for_comparison(base_scenario, "optimistic"),
        "Crisis": modify_scenario_for_comparison(base_scenario, "crisis")
    }
    
    results_comparison = {}
    
    # Ejecutar simulaciones para cada escenario
    for scenario_name, scenario_data in scenarios.items():
        print(f"\nEjecutando simulación: {scenario_name}")
        results = model.run_simulation(scenario_data)
        results_comparison[scenario_name] = results
        
        print(f"  VPN: {format_currency(results.npv)}")
        print(f"  TIR: {results.irr:.2%}" if not pd.isna(results.irr) else "  TIR: No calculable")
    
    # Mostrar comparación final
    print("\n" + "-" * 60)
    print("RESUMEN COMPARATIVO")
    print("-" * 60)
    
    for scenario_name, results in results_comparison.items():
        viability = "✓" if results.npv > 0 else "✗"
        print(f"{scenario_name:12} | VPN: {results.npv:>12,.0f} | {viability}")


def modify_scenario_for_comparison(base_scenario: dict, scenario_type: str) -> dict:
    """
    Modifica el escenario base para crear variaciones comparativas.
    
    Args:
        base_scenario (dict): Escenario base para modificar
        scenario_type (str): Tipo de escenario ("conservative", "optimistic", "crisis")
        
    Returns:
        dict: Escenario modificado
    """
    modified = base_scenario.copy()
    
    if scenario_type == "conservative":
        # Reducir ingresos 20% y aumentar costos 10%
        modified["revenue_projections"] = [r * 0.8 for r in base_scenario["revenue_projections"]]
        modified["cost_projections"] = [c * 1.1 for c in base_scenario["cost_projections"]]
        
    elif scenario_type == "optimistic":
        # Aumentar ingresos 30% y reducir costos 5%
        modified["revenue_projections"] = [r * 1.3 for r in base_scenario["revenue_projections"]]
        modified["cost_projections"] = [c * 0.95 for c in base_scenario["cost_projections"]]
        
    elif scenario_type == "crisis":
        # Reducir ingresos 40% y aumentar costos 20%
        modified["revenue_projections"] = [r * 0.6 for r in base_scenario["revenue_projections"]]
        modified["cost_projections"] = [c * 1.2 for c in base_scenario["cost_projections"]]
    
    return modified


if __name__ == "__main__":
    """
    Punto de entrada principal para la ejecución del ejemplo.
    
    Ejecuta la simulación básica y opcionalmente el análisis comparativo
    dependiendo de los argumentos de línea de comandos proporcionados.
    """
    import pandas as pd
    
    try:
        # Ejecutar simulación básica
        run_basic_simulation()
        
        # Preguntar si ejecutar análisis comparativo
        print("\n¿Desea ejecutar el análisis comparativo de escenarios? (s/n): ", end="")
        user_input = input().lower().strip()
        
        if user_input in ['s', 'si', 'sí', 'y', 'yes']:
            run_comparative_scenarios()
        
        print("\nEjemplo completado exitosamente.")
        
    except KeyboardInterrupt:
        print("\n\nEjecución interrumpida por el usuario.")
    except Exception as e:
        print(f"\nError durante la ejecución: {str(e)}")
        print("Consulte la documentación o contacte al equipo de soporte.")
        raise