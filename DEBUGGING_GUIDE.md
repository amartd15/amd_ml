# Guía de depuración en VS Code para proyectos C/C++, CUDA y librerías .so

Esta guía explica cómo preparar un proyecto para depurar correctamente en Visual Studio Code, incluyendo casos con CUDA, CMake, ejecutables y librerías compartidas (`.so`).

## 1. Objetivo

El objetivo es que puedas:
- ejecutar tu programa desde VS Code en modo depuración,
- detenerte en breakpoints en archivos fuente,
- inspeccionar variables y stack traces,
- trabajar con proyectos que usen CMake, CUDA y librerías dinámicas.

## 2. Requisitos previos

Asegúrate de tener instalados:
- un compilador de C/C++ (por ejemplo `g++` / `gcc`),
- `gdb` si vas a depurar código nativo,
- `nvcc` y `cuda-gdb` si trabajas con CUDA.

### Comprobar instalación

```bash
which gcc
which g++
which gdb
which nvcc
which cuda-gdb
```

## 3. Compilar en modo Debug

Para que el depurador pueda mostrar información útil, el proyecto debe compilar con símbolos de depuración.

### Con CMake

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug
cmake --build build
```

### Para proyectos CUDA

Si usas NVCC, además conviene configurar el modo debug con:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc
cmake --build build
```

Y en CMake puedes añadir algo como:

```cmake
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -G -g")
```

Esto genera información de depuración que el debugger puede leer.

## 4. Configurar VS Code

VS Code necesita dos archivos principales:
- `.vscode/launch.json`: define cómo se lanza el programa en depuración.
- `.vscode/settings.json`: define opciones globales del entorno, como el tipo de build.

## 5. Configuración básica de launch.json

Un ejemplo simple para depurar un ejecutable nativo:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug App",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/build/mi_programa",
      "cwd": "${workspaceFolder}",
      "environment": [
        {
          "name": "LD_LIBRARY_PATH",
          "value": "${workspaceFolder}/build:${env:LD_LIBRARY_PATH}"
        }
      ],
      "MIMode": "gdb",
      "miDebuggerPath": "/usr/bin/gdb"
    }
  ]
}
```

### Explicación

- `program`: ruta del ejecutable que se quiere depurar.
- `cwd`: directorio de trabajo desde el que se ejecuta el programa.
- `environment`: permite añadir rutas como `LD_LIBRARY_PATH` para localizar librerías `.so`.
- `MIMode`: indica qué depurador usar (`gdb`).

## 6. Configuración para proyectos CUDA

Si tu proyecto usa CUDA, el launch configuration debe apuntar al ejecutable generado por `nvcc` y usar `cuda-gdb`.

Ejemplo:

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "CUDA Debug",
      "type": "cuda-gdb",
      "request": "launch",
      "program": "${workspaceFolder}/build/mi_programa",
      "cwd": "${workspaceFolder}",
      "environment": [
        {
          "name": "LD_LIBRARY_PATH",
          "value": "${workspaceFolder}/build:${env:LD_LIBRARY_PATH}"
        }
      ],
      "miDebuggerPath": "/usr/local/cuda-12.5/bin/cuda-gdb",
      "setupCommands": [
        {
          "description": "Enable pretty-printing",
          "text": "set print pretty on",
          "ignoreFailures": true
        },
        {
          "description": "Allow pending breakpoints",
          "text": "set breakpoint pending on",
          "ignoreFailures": true
        }
      ]
    }
  ]
}
```

## 7. Importante: cómo depurar con librerías `.so`

Si tu programa depende de bibliotecas compartidas, el depurador debe saber dónde buscarlas.

### Solución

Añade esta variable de entorno en `launch.json`:

```json
"environment": [
  {
    "name": "LD_LIBRARY_PATH",
    "value": "${workspaceFolder}/build:${env:LD_LIBRARY_PATH}"
  }
]
```

Esto permite que el sistema encuentre archivos como:
- `libmi_libreria.so`
- `libamd_ml_lr_lib.so`

Sin esta configuración, el programa puede ejecutarse mal o el depurador puede no encontrar símbolos correctamente.

## 8. Configuración recomendada para CMake en VS Code

En `.vscode/settings.json` puedes dejar algo como esto:

```json
{
  "cmake.buildType": "Debug",
  "cmake.configureArgs": [
    "-DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.5/bin/nvcc"
  ]
}
```

Esto hace que los proyectos basados en CMake se construyan automáticamente en modo Debug.

## 9. Cómo usar breakpoints correctamente

1. Abre el archivo fuente que quieres depurar.
2. Haz clic al lado del número de línea.
3. Inicia el debug con F5.
4. Si el breakpoint no se alcanza, revisa:
   - que el programa se compiló con `-g` o `-G`,
   - que el ejecutable corresponde al código fuente que estás viendo,
   - que la ruta del ejecutable en `launch.json` es correcta,
   - que la librería `.so` está disponible.

## 10. Problemas comunes

### El breakpoint aparece como hueco

Esto suele indicar que el depurador no tiene acceso a los símbolos o el código no coincide con el ejecutable actual.

### El programa no encuentra una librería `.so`

Revisa `LD_LIBRARY_PATH` y la ruta real de la librería en el árbol de build.

### El depurador no entra en funciones CUDA

Comprueba:
- que estás usando `cuda-gdb` en vez de `gdb` clásico,
- que el programa se compiló con símbolos de depuración,
- que el archivo `.cu` está correctamente asociado al compilador CUDA.

## 11. Plantilla reutilizable para futuro

### `.vscode/launch.json`

```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "Debug",
      "type": "cppdbg",
      "request": "launch",
      "program": "${workspaceFolder}/build/mi_programa",
      "cwd": "${workspaceFolder}",
      "environment": [
        {
          "name": "LD_LIBRARY_PATH",
          "value": "${workspaceFolder}/build:${env:LD_LIBRARY_PATH}"
        }
      ],
      "MIMode": "gdb",
      "miDebuggerPath": "/usr/bin/gdb"
    }
  ]
}
```

### `.vscode/settings.json`

```json
{
  "cmake.buildType": "Debug"
}
```

## 12. Resumen rápido

Para cualquier proyecto futuro, recuerda:
1. compilar con Debug,
2. definir un `launch.json` correcto,
3. incluir `LD_LIBRARY_PATH` si usas `.so`,
4. usar `cuda-gdb` si trabajas con CUDA,
5. verificar que el ejecutable y los archivos fuente coinciden.

Con estos pasos, tendrás un flujo de depuración sólido y reutilizable para casi cualquier proyecto en VS Code.
