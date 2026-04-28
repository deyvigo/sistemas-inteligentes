# Título del proyecto
Generación de Pictogramas con Auto-Mejora mediante LLM-Judge y Retroalimentación Humana

# Planteamiento del Problema
¿Cómo diseñar un sistema que mejore iterativamente la generación de pictogramas
utilizando evaluación automática (LLM-Judge) y retroalimentación humana?

# Objetivos
Desarrollar un sistema de generación de pictogramas que se auto-mejore iterativamente mediante: evaluación automática con LLM (judge), correcciones humanas estructuradas

## Objetivos específicos
- Implementar un sistema base de generación de pictogramas.
- Diseñar un módulo de evaluación automática basado en LLM (LLM-Judge).
- Implementar un sistema de retroalimentación humana (human-in-the-loop).
- Diseñar un mecanismo que convierta críticas (LLM + humano) en mejoras del sistema.
- Evaluar el impacto del sistema en:
  - calidad semántica
  - reducción de errores
  - eficiencia del proceso

# ¿Cómo hacer?

## Fase 1: Sistema base de generación
- Entrada: Texto en español
- Salida: Secuencia de pictogramas (IDs ARASAAC)
- Métodos mínimos: Sistema basado en reglas o sistema basado en extracción de conceptos (LLM o heurísticas)

## Fase 2: Módulo LLM-Judge
- Requisitos: El LLM debe analizar (cobertura semántica, errores de selección, orden de pictogramas)
- Salida estructurada obligatoria:
```json
{
  "score": 1-5,
  "missing_concepts": [...],
  "incorrect_pictograms": [...],
  "ordering_issues": [...],
  "suggestions": [...]
}
```

## Fase 3: Interfaz web (Human-in-the-loop)

Funcionalidades obligatorias:
### 1. Visualización
- Texto original
- Pictogramas generados

### 2. Evaluación del LLM
- Mostrar score, errores detectados, sugerencias

### 3. Corrección humana
- Permitir (eliminar pictogramas, agregar pictogramas, reordenar secuencia)

### 4. Registro de feedback
- Cada iteración debe almacenarse como:
```json
{
  "texto": "...",
  "prediccion": [...],
  "judge_output": {...},
  "correccion_humana": [...],
  "acciones": [...]
}
```

## Fase 4: Módulo de conversión de feedback
Se debe transformar críticas de LLM, correcciones humanas en mejoras del sistema.
### Estrategias a implementar (mínimo 2)

#### 1. Refinamiento de salida (post-procesamiento)
- Aplica sugerencias del LLM directamente
#### 2. Mejora de reglas
- Ajustar mapeos incorrectos, añadir excepciones
#### 3. Refinamiento de prompts
- Adaptar instrucciones según errores recurrentes

## Fase 5: Loop de auto-mejora
- El sistema debe implementar el siguiente ciclo:
Generar → Evaluar (LLM) → Corregir (Humano) → Mejorar → Re-generar

## Fase 6: Evaluación experimental