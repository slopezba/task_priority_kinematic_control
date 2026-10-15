# task_priority_kinematic_control

Paquete ROS 2 para control cinemático de cuerpo completo con jerarquía de tareas. Está pensado para CIRTESUB con dos manipuladores Alpha y calcula velocidades para:

- la base del vehículo;
- el brazo Alpha izquierdo;
- el brazo Alpha derecho.

La idea central es resolver varias tareas cinemáticas ordenadas por prioridad. Las tareas de prioridad alta se intentan cumplir primero y las de prioridad baja se proyectan en el espacio nulo de las anteriores para no interferir, siempre que sea posible.

## Qué contiene

- Un controlador `ros2_control`: `task_priority_kinematic_control/TaskPriorityController`.
- Un nodo ejecutable independiente: `task_priority_runtime`.
- Plugins de cinemática: KDL y Pinocchio.
- Plugins de tareas: pose, posición, orientación, límites articulares, postura nominal y yaw de base.
- Mensajes y servicios para inspección y gestión en runtime.
- Configuraciones de ejemplo para nodo normal y para `ros2_control`.

## Modos de uso

### Nodo `task_priority_runtime`

Ejecutable independiente definido en `src/main.cpp` y configurado por `config/task_priority_kinematic_control.yaml`.

Este modo:

- se suscribe a navegación y `joint_states`;
- publica comandos de velocidad para base y brazos en topics;
- publica diagnóstico del solver;
- permite cambiar backend con servicio;
- ejecuta el lazo con `rate_hz`.

Ejemplo:

```bash
ros2 launch task_priority_kinematic_control task_priority_kinematic_control.launch.py \
  params_file:=/ruta/a/task_priority_kinematic_control.yaml
```

### Controlador `ros2_control`

Plugin definido en `task_priority_kinematic_control_plugins.xml`:

```text
task_priority_kinematic_control/TaskPriorityController
```

Este modo está pensado para cargarse desde `controller_manager`, usando `config/task_priority_controller_ros2_control.yaml`.

El controlador:

- toma estados articulares desde interfaces de estado `position` y `velocity`;
- toma el estado de navegación desde un topic;
- escribe la velocidad de base en interfaces de comando `body_velocity/*`;
- publica comandos de velocidad de los brazos en topics `Float64MultiArray`;
- publica estado global de jerarquía y estado individual por tarea.

## Flujo de datos

1. El sistema recibe estado de base desde `sura_msgs/msg/Navigator`.
2. Recibe estado articular de los dos Alpha.
3. El backend cinemático calcula poses y jacobianos de los frames.
4. Cada task calcula:
   - error;
   - velocidad deseada de la tarea;
   - jacobiano;
   - estado activo/inactivo.
5. `HierarchySolver` resuelve las tareas por prioridad.
6. Se saturan las velocidades con `velocity_limits`.
7. Se separa el vector de velocidad generalizado en base, brazo izquierdo y brazo derecho.

El orden del vector de velocidad generalizado es:

```text
[base_dofs_activos..., left_arm_joints..., right_arm_joints...]
```

Con la configuración actual:

```text
[base x, base y, base z, base roll, base pitch, base yaw,
 alpha_left axis_a..axis_e,
 alpha_right axis_a..axis_e]
```

## Parámetros globales

Estos parámetros aparecen tanto en el nodo runtime como en el controlador `ros2_control`, salvo donde se indique lo contrario.

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `backend_plugin` | string | Plugin de cinemática. Por defecto `task_priority_kinematic_control/KDLKinematicsBackend`. |
| `robot_description` | string | URDF completo. Si está vacío se busca en otro nodo. |
| `robot_description_source_node` | string | Nodo del que se lee el parámetro `robot_description`. En los ejemplos: `/robot_state_publisher_cirtesub`. |
| `robot_description_wait_timeout_sec` | double | Timeout para esperar el servicio de parámetros del `robot_description_source_node`. |
| `world_frame` | string | Frame mundo usado como referencia. En los ejemplos: `world_ned`. |
| `base_frame` | string | Frame base del vehículo. |
| `left_tip_frame` | string | Frame herramienta del Alpha izquierdo. |
| `right_tip_frame` | string | Frame herramienta del Alpha derecho. |
| `left_arm_joints` | string[] | Joints del brazo izquierdo, en el orden usado por el solver. |
| `right_arm_joints` | string[] | Joints del brazo derecho, en el orden usado por el solver. |
| `active_base_dofs` | int[] | DOFs de base controlados. Valores válidos: `0..5`. |
| `solver_method` | string | Método del inversor: `dls`, `pinv` o `svd`. |
| `dls_lambda` | double | Amortiguamiento del método DLS. Debe ser positivo. |
| `dof_weights` | double[] | Pesos por DOF. Mayor peso penaliza más ese DOF. |
| `velocity_limits` | double[] | Límite absoluto de velocidad por DOF del vector generalizado. |
| `navigator_topic` | string | Topic de entrada con `sura_msgs/msg/Navigator`. |
| `task_ids` | string[] | Lista de tareas que se cargan mediante pluginlib. |

Solo en `task_priority_runtime`:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `rate_hz` | double | Frecuencia del timer del nodo. Mínimo interno: `1.0 Hz`. |
| `joint_states_topic` | string | Topic de entrada con `sensor_msgs/msg/JointState`. |
| `base_command_topic` | string | Topic de salida para comando de base `geometry_msgs/msg/TwistStamped`. |

Solo en `TaskPriorityController`:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `body_velocity_controller_name` | string | Nombre base de las 6 interfaces de comando de la base. Por defecto `body_velocity`. |

En ambos modos:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `left_arm_command_topic` | string | Topic de salida `Float64MultiArray` para velocidades del brazo izquierdo. |
| `right_arm_command_topic` | string | Topic de salida `Float64MultiArray` para velocidades del brazo derecho. |

## DOFs de base

`active_base_dofs` selecciona qué grados de libertad de la base entran en el solver:

| Índice | DOF |
| --- | --- |
| `0` | `linear.x` |
| `1` | `linear.y` |
| `2` | `linear.z` |
| `3` | `angular.x` |
| `4` | `angular.y` |
| `5` | `angular.z` |

Si usas `[0, 1, 2, 5]`, por ejemplo, el solver solo controlará traslación XYZ y velocidad angular Z en ejes del mundo. Los DOFs no activos se rellenan con cero antes de convertir la velocidad a ejes de la base; esta rotación puede generar componentes en otros ejes de la base. El controlador `ros2_control` siempre envía las 6 componentes del comando de base.

## Solver jerárquico

El solver está en `src/core/hierarchy_solver.cpp`.

Para cada tarea activa:

1. Aplica el proyector del espacio nulo acumulado.
2. Calcula una inversa ponderada del jacobiano.
3. Suma la corrección de velocidad sin destruir tareas anteriores.
4. Actualiza el proyector.

Métodos disponibles:

- `dls`: damped least squares. Es el valor recomendado para operación normal.
- `pinv`: pseudoinversa mediante descomposición ortogonal completa.
- `svd`: pseudoinversa basada en SVD.

`dof_weights` se usa como métrica de ponderación. `velocity_limits` satura al final cada elemento del vector de velocidad generalizado.

## Plugins de cinemática

### `task_priority_kinematic_control/KDLKinematicsBackend`

Backend funcional actual.

Hace lo siguiente:

- parsea `robot_description` con `kdl_parser`;
- construye cadenas desde `base_frame` hasta los frames disponibles;
- calcula FK y jacobianos con KDL;
- añade la contribución de la base al jacobiano del efector;
- cachea el estado de cada frame en cada ciclo.

Nombre reportado en mensajes: `kdl`.

### `task_priority_kinematic_control/PinocchioKinematicsBackend`

Plugin registrado, pero actualmente es un placeholder.

Configura el modelo y reporta nombre `pinocchio`, pero `get_frame_state()` y `get_relative_transform()` lanzan excepción porque aún no están implementados. Para ejecución real usa KDL.

## Plugins de tareas

Todas las tareas comparten estos parámetros:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `tasks.<id>.plugin` | string | Nombre del plugin de tarea. |
| `tasks.<id>.enabled` | bool | Activa o desactiva la tarea al arrancar. |
| `tasks.<id>.priority` | double | Prioridad inicial. Menor número significa mayor prioridad. Internamente se convierte a `uint32`. |
| `tasks.<id>.group` | string | Etiqueta lógica: `left`, `right`, `base`, `global`, etc. |

Las tareas se cargan siguiendo `task_ids` y luego se ordenan por `priority`.

### `EndEffectorPoseTask`

Plugin:

```text
task_priority_kinematic_control/EndEffectorPoseTask
```

Controla pose cartesiana completa de un efector o frame: posición XYZ y orientación.

Parámetros:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `frame_id` | string | Frame a controlar. Si no se define, usa `left_tip_frame`. |
| `gain` | double[6] | Ganancias `[x, y, z, roll, pitch, yaw]` sobre el error de pose. |

Entrada runtime:

```text
/cirtesub/controller/task_priority/tasks/<id>/target
geometry_msgs/msg/PoseStamped
```

Si no se recibe objetivo, la tarea mantiene la posición actual y no genera error de orientación.

Salida interna:

- error de tamaño 6;
- jacobiano de tamaño `6 x total_dofs`;
- velocidad deseada `gain * error`.

### `EndEffectorPositionTask`

Plugin:

```text
task_priority_kinematic_control/EndEffectorPositionTask
```

Controla solo la posición XYZ de un frame.

Parámetros:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `frame_id` | string | Frame a controlar. |
| `gain` | double[3] | Ganancias de posición XYZ. |
| `default_goal` | double[3] | Objetivo usado si no llega un `PoseStamped`. |

Entrada runtime:

```text
/cirtesub/controller/task_priority/tasks/<id>/target
geometry_msgs/msg/PoseStamped
```

Solo usa `pose.position`; ignora la orientación.

### `EndEffectorOrientationTask`

Plugin:

```text
task_priority_kinematic_control/EndEffectorOrientationTask
```

Controla solo la orientación de un frame.

Parámetros:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `frame_id` | string | Frame a controlar. |
| `gain` | double[3] | Ganancias del error rotacional. |

Entrada runtime:

```text
/cirtesub/controller/task_priority/tasks/<id>/target
geometry_msgs/msg/PoseStamped
```

Solo usa `pose.orientation`. Si no hay objetivo, usa orientación identidad como referencia.

### `FramePoseTask`

Plugin:

```text
task_priority_kinematic_control/FramePoseTask
```

Controla los ejes seleccionados de la pose de un frame arbitrario. También puede controlar la base si `frame_id == base_frame`.

Parámetros:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `frame_id` | string | Frame a controlar. Si no se define, usa `base_frame`. |
| `gain` | double[6] | Ganancias de pose completa. |
| `activation` | int[6] | Máscara 0/1 en orden x, y, z, rotación X, rotación Y, rotación Z. Omitida o vacía activa los seis ejes. |
| `feedforward_timeout` | double | Caducidad en segundos; 0 mantiene el comportamiento sin caducidad. Se configura al iniciar. |
| `roll_pitch_control` | bool | Activa la política de inclinación para la base con máscara `[0,0,0,1,1,0]`; false por defecto. |
| `allow_roll_pitch` | bool | Permiso inicial de inclinación; false fija roll/pitch a cero cuando la política está activa. |

Entrada runtime:

```text
/cirtesub/controller/task_priority/tasks/<id>/target
geometry_msgs/msg/PoseStamped
```

Si controla la base, usa directamente `Navigator.position` y `Navigator.rpy`. Si controla otro frame, usa el backend cinemático.

Los ejes de la máscara están expresados en el frame mundo; las componentes
angulares son las del error de orientación y de la velocidad angular, no tasas
de Euler. Los ejes excluidos se eliminan del Jacobiano y su feedforward se ignora,
también al evaluar el umbral de actualización automática del objetivo.
La configuración dual de CIRTESUB usa `stabilize` con `[0, 0, 0, 1, 1, 0]`
y prioridad 3, y `base_pose` con `[1, 1, 1, 0, 0, 1]` y prioridad 6.

Con `roll_pitch_control: true`, se usan errores de roll/pitch y el Jacobiano de
sus tasas a partir de la velocidad angular en mundo. Yaw queda libre y no cambia
los setpoints de inclinación. Los servicios `std_srvs/Trigger`
`<prefijo de tareas>/<id>/enable_roll_pitch` y `disable_roll_pitch` permiten o
bloquean inclinaciones sin deshabilitar la tarea. Al permitirlas, el objetivo
sigue el movimiento y captura la inclinación medida en el primer ciclo con
velocidad cero o caducada. Al bloquearlas vuelve a cero. Los objetivos de pose
externos no sustituyen esta referencia gestionada por la política.
La configuración dual usa `feedforward_timeout: 0.5` en `stabilize`.
La singularidad de Euler próxima a pitch ±90° desactiva el cálculo de esta tarea.

### `StabilizeLimitsTask`

Plugin: `task_priority_kinematic_control/StabilizeLimitsTask`.

Protege los límites de roll/pitch de la base mediante restricciones que se
activan cerca de los límites y se liberan con histéresis. Es independiente de
`stabilize`: no recibe objetivos de pose ni feedforward y no tiene servicios
`enable_roll_pitch`/`disable_roll_pitch`. Usa los servicios generales de tareas
para habilitación, deshabilitación y prioridad. La configuración dual de
CIRTESUB la registra como `stabilize_limits`, con prioridad 2 y apagada al inicio.
El teleop conserva su comportamiento y no cambia el estado de esta tarea.

Todos los límites y ajustes se leen del YAML al configurar el controlador:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `lower_limits` | double[2] | Límites inferiores `[roll, pitch]`, en radianes. Obligatorio. |
| `upper_limits` | double[2] | Límites superiores `[roll, pitch]`, en radianes. Obligatorio. |
| `alpha` | double | Distancia al límite para activar la restricción; valor inicial 0.06981317 rad (4°). |
| `delta` | double | Distancia al límite para liberar la restricción; valor inicial 0.13962634 rad (8°). |
| `eps` | double | Tolerancia de liberación; valor inicial 0.0001 rad. |
| `gain_scalar` | double | Ganancia correctora no negativa; valor inicial 2.0. |
| `max_repulsive_velocity` | double | Máxima tasa correctora de roll/pitch, en rad/s; valor inicial 0.3. |
| `debug` | bool | Muestrear y publicar los indicadores solamente en modo debug; false por defecto, true en el perfil dual. |
| `publish_marker_rate` | double | Frecuencia de muestreo y publicación de markers en Hz; 2 por defecto y en el perfil dual. Requiere reconfigurar para cambiarla. |
| `enabled`, `priority`, `group` | bool, double, string | Ajustes comunes de la tarea. |

Los márgenes son comunes a ambos ejes. Debe cumplirse `0 <= alpha < delta`,
`0 <= eps < delta-alpha`, y cada intervalo debe ser mayor que `2*delta`.
Los límites de pitch deben estar estrictamente entre -π/2 y π/2 para evitar
incluir la singularidad de Euler. Todos los valores numéricos deben ser finitos.

Con límites de ±20°, se activa a ±16° y se libera aproximadamente a ±12°.
La velocidad correctora apunta a `lower+delta` o `upper-delta`, limitada por
`max_repulsive_velocity`. Cada eje se activa por separado; las filas inactivas
del Jacobiano son cero. Yaw, traslaciones y articulaciones quedan libres de esta
restricción. Se usa el Jacobiano de tasas de roll/pitch respecto a la velocidad
angular en mundo, teniendo en cuenta el yaw actual.

Deshabilitar o reiniciar borra la histéresis. Navegación ausente, valores no
finitos o una singularidad de pitch producen una tarea inactiva con diagnóstico.
Estas restricciones siguen el mecanismo de activación del solver existente.

Los indicadores se publican como `visualization_msgs/MarkerArray` en
`<nombre completo del nodo controlador>/stabilize_limits/limits_markers`.
Para el perfil dual, el tópico es:

```text
/cirtesub/alpha/controller/task_priority_controller/stabilize_limits/limits_markers
```

El perfil `cirtesub.rviz` incluye el display **Stabilize limits**. Se muestran
dos indicadores sin texto junto a la base, con arco de límites, bandas de
activación, marcas perpendiculares y flecha del ángulo actual. El indicador de
roll está centrado lateralmente en `[0, 0, -0.15]` de la base; el de pitch está en
`[0, 0.22, -0.18]` metros, en el mismo frame.
Cada arco y su flecha comparten centro; el radio del arco es 0.18 m y la flecha
mide 0.16 m. Las marcas perpendiculares del límite son rojas; las del umbral
de activación (`lower+alpha`, `upper-alpha`) son naranjas. El arco, las bandas
y la flecha adoptan el color del estado de su eje:

- Verde: eje libre, a más de `delta` del límite.
- Naranja: cerca del límite, dentro de `delta`, pero restricción inactiva.
- Rojo: restricción actuando, incluyendo el estado conservado por histéresis.
- Gris: tarea deshabilitada; todos los elementos, incluidas las marcas
  perpendiculares, siguen visibles en gris con debug activo y datos válidos.

La visualización usa tres snapshots fijos sin bloqueos. Con `debug: false`
no se generan snapshots; habilitado, se muestrean a `publish_marker_rate`
usando la marca temporal del control. Los markers y sus vectores de puntos se
preasignan al crear el observador; la geometría de límites se calcula una vez
y se reutiliza. Solo se actualizan colores, marcas temporales y extremos de
flechas. La creación y publicación ROS/DDS quedan en el hilo del observador,
fuera del ciclo de control. Esta separación no garantiza por sí sola tiempo
real estricto del controlador completo.

El parámetro `debug` se puede alternar en ejecución. Al apagarlo se envía una
eliminación de los indicadores visibles; también se borran ante navegación
inválida o si los snapshots superan `max(1, 2/publish_marker_rate)` segundos de
antigüedad. Los markers caducan tras `max(2, 3/publish_marker_rate)` segundos
como respaldo ante una interrupción del publicador (2 s a 2 Hz). La
visualización no habilita la tarea ni cambia las restricciones o prioridades.

La jerarquía de la configuración dual es: `joint_limits` (0),
`self_collision_avoidance` (1), `stabilize_limits` (2), `stabilize` (3),
`right_pose` (4), `left_pose` (5), `base_pose` (6), `left_joint_trajectory` (7),
`right_joint_trajectory` (8), `bimanual_joint_trajectory` (9),
`left_arm_nominal` (10), `right_arm_nominal` (11), `EndEffectors_Relative_pose`
(12), `ee_plan_relative_pose` (13) y `base_yaw` (14).

### `JointNominalTask`

Plugin:

```text
task_priority_kinematic_control/JointNominalTask
```

Empuja un conjunto de articulaciones hacia una configuración nominal. Es útil como tarea secundaria para evitar posturas raras o mantener el brazo cerca de una zona cómoda.

También recibe `sensor_msgs/JointState` en
`<prefijo de tareas>/<id>/joint_command`, con todos los nombres configurados,
posiciones objetivo y velocidades de feedforward. Se aceptan nombres en distinto
orden y se aplican posición y velocidad juntas bajo el bloqueo del host.
El comando es `v_ff + gain * (objetivo - posición)` en los ejes activos.
Mensajes incompletos, duplicados o no finitos borran las velocidades y conservan
el objetivo anterior. Deshabilitar, reiniciar o caducar también borra velocidades.
`joint_command_timeout` vale 0,5 s por defecto y debe ser positivo.
La entrada `joint_target` sigue disponible y borra el feedforward al aceptarse.

Parámetros:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `joint_names` | string[] | Joints controlados por esta tarea. |
| `target` | double[] | Posición articular objetivo. Debe corresponder con `joint_names`. |
| `gain` | double[] | Ganancia por joint. |

Entrada runtime:

```text
/cirtesub/controller/task_priority/tasks/<id>/joint_target
std_msgs/msg/Float64MultiArray
```

El tamaño de `data` debe coincidir exactamente con `joint_names`.

Salida interna:

- error `target - joint_position`;
- jacobiano identidad sobre las columnas de esos joints;
- velocidad deseada `gain * error`.

### `JointLimitsTask`

Plugin:

```text
task_priority_kinematic_control/JointLimitsTask
```

Genera una velocidad de repulsión cuando una articulación entra en la zona cercana a sus límites.

Parámetros:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `lower_limits` | double[] | Límites inferiores de todos los joints del modelo, en orden izquierda + derecha. |
| `upper_limits` | double[] | Límites superiores de todos los joints del modelo, en orden izquierda + derecha. |
| `margin` | double | Anchura de la zona de seguridad cerca de cada límite. |
| `gain_scalar` | double | Ganancia de la velocidad de repulsión. |

No recibe objetivos por topic.

Comportamiento:

- si `q < lower + margin`, manda velocidad positiva;
- si `q > upper - margin`, manda velocidad negativa;
- si está dentro de la zona segura, manda cero.

### `BaseYawTask`

Plugin:

```text
task_priority_kinematic_control/BaseYawTask
```

Controla el yaw de la base usando `Navigator.rpy.z`.

Parámetros:

| Parámetro | Tipo | Descripción |
| --- | --- | --- |
| `target_yaw` | double | Yaw objetivo en radianes. |
| `gain_scalar` | double | Ganancia proporcional. |

No recibe objetivos por topic en la implementación actual. El error angular se normaliza a `[-pi, pi]`.

Importante: para que esta tarea tenga efecto, `active_base_dofs` debe incluir el DOF `5` (`angular.z`).

## Configuración de tareas de ejemplo

La configuración incluida carga esta jerarquía:

| Prioridad | ID | Plugin | Grupo | Función |
| --- | --- | --- | --- | --- |
| `0` | `left_pose` | `EndEffectorPoseTask` | `left` | Controla pose del efector izquierdo. |
| `1` | `right_pose` | `EndEffectorPoseTask` | `right` | Controla pose del efector derecho. |
| `2` | `joint_limits` | `JointLimitsTask` | `global` | Evita límites articulares. |
| `3` | `left_arm_nominal` | `JointNominalTask` | `left` | Postura nominal del brazo izquierdo. |
| `4` | `right_arm_nominal` | `JointNominalTask` | `right` | Postura nominal del brazo derecho. |
| `5` | `base_yaw` | `BaseYawTask` | `base` | Mantiene yaw de base. |

La tarea `joint_limits` utiliza buffers preasignados y puede mostrar cilindros magenta sobre las
articulaciones con límites activos. Los parámetros y el comportamiento se describen en
[docs/joint_limits.md](docs/joint_limits.md).

## Topics de entrada

### Estado de navegación

```text
Topic por defecto: /cirtesub/navigator/navigation
Tipo: sura_msgs/msg/Navigator
```

Campos usados:

- `position.position.x/y/z`;
- `rpy.x/y/z`.

### Estado articular en `task_priority_runtime`

```text
Topic por defecto: /cirtesub/alpha/joint_states
Tipo: sensor_msgs/msg/JointState
```

Usa `name`, `position` y `velocity`. Los nombres deben coincidir con `left_arm_joints` y `right_arm_joints`.

### Estado articular en `TaskPriorityController`

El controlador no se suscribe a `joint_states`. Lee interfaces de estado:

```text
<joint>/position
<joint>/velocity
```

para todos los joints de `left_arm_joints` y `right_arm_joints`.

### Objetivos de pose

Para tareas cuyo nombre de plugin contiene `PoseTask`:

```text
/cirtesub/controller/task_priority/tasks/<task_id>/target
geometry_msgs/msg/PoseStamped
```

Esto incluye:

- `EndEffectorPoseTask`;
- `FramePoseTask`;

y, por la lógica actual del código, cualquier plugin cuyo nombre contenga `PoseTask`.

### Objetivos articulares nominales

Para `JointNominalTask`:

```text
/cirtesub/controller/task_priority/tasks/<task_id>/joint_target
std_msgs/msg/Float64MultiArray
```

## Topics de salida

### Comando de base en `task_priority_runtime`

```text
Topic por defecto: /cirtesub/controller/task_priority/base_cmd
Tipo: geometry_msgs/msg/TwistStamped
```

Mapea el vector de base a:

- `twist.linear.x/y/z`;
- `twist.angular.x/y/z`.

### Comando de base en `TaskPriorityController`

El solver calcula la velocidad lineal y angular de la base en `world_frame`.
Antes de enviarla a `body_velocity`, el controlador rota ambas componentes a
`base_frame` usando la orientación RPY actual del navegador:
`v_base = R_world_base.transpose() * v_world` y
`omega_base = R_world_base.transpose() * omega_world`.
La rotación sigue la misma convención `Rz(yaw) * Ry(pitch) * Rx(roll)` que la
cinemática. El punto de referencia sigue siendo el origen de la base.

Con `base_command_mode: topic`, publica `geometry_msgs/msg/Twist` en
`base_command_topic`. Con `base_command_mode: interfaces`, escribe las interfaces:

```text
<body_velocity_controller_name>/linear.x
<body_velocity_controller_name>/linear.y
<body_velocity_controller_name>/linear.z
<body_velocity_controller_name>/angular.x
<body_velocity_controller_name>/angular.y
<body_velocity_controller_name>/angular.z
```

Con la configuración por defecto:

```text
body_velocity/linear.x
body_velocity/linear.y
body_velocity/linear.z
body_velocity/angular.x
body_velocity/angular.y
body_velocity/angular.z
```

### Comandos de brazos

En ambos modos:

```text
/cirtesub/alpha/controller/alpha_left_forward_velocity_controller/commands
std_msgs/msg/Float64MultiArray

/cirtesub/alpha/controller/alpha_right_forward_velocity_controller/commands
std_msgs/msg/Float64MultiArray
```

El orden del array coincide con `left_arm_joints` o `right_arm_joints`.

### Output total del controlador

Solo en `TaskPriorityController`:

```text
Topic relativo: task_priority/output
Tipo: task_priority_kinematic_control/msg/ControllerOutput
```

Si el controlador está en el namespace `/cirtesub/controller`, ROS 2 lo resuelve como:

```text
/cirtesub/controller/task_priority/output
```

Publica en un solo mensaje:

- velocidad generalizada completa;
- nombres de cada elemento de la velocidad generalizada;
- velocidad 6D de la base en ejes de `world_frame`, en orden `linear.x/y/z`, `angular.x/y/z`, antes de convertirla a ejes de la base para `body_velocity`;
- velocidades del brazo izquierdo y nombres de joints;
- velocidades del brazo derecho y nombres de joints.

### Estado de la jerarquía

```text
Topic: /hierarchy_state
Tipo: task_priority_kinematic_control/msg/HierarchyState
```

Contenido:

- `backend_name`;
- `solver_method`;
- `ready`;
- lista de `TaskStatus`.

En el runtime, `ready` es `true` solo cuando ya llegaron navegación y joints. En el controlador se publica como `true` durante `update()`.

### Diagnóstico del solver

Solo en `task_priority_runtime`:

```text
Topic: /solver_diagnostics
Tipo: task_priority_kinematic_control/msg/SolverDiagnostics
```

Contenido:

- backend usado;
- método del solver;
- `damped_least_squares_lambda`;
- estimación de condición;
- `state_valid`;
- mensaje `running` o `waiting_for_state`.

### Estado individual por tarea

Solo en `TaskPriorityController`:

```text
Topic relativo: task_priority/tasks/<task_id>/state
task_priority_kinematic_control/msg/TaskState
```

Si el controlador está en `/cirtesub/controller`, queda:

```text
/cirtesub/controller/task_priority/tasks/<task_id>/state
```

Publica:

- `id`;
- `type`;
- `enabled`;
- `active`;
- `has_frame_pose`;
- `frame_id`;
- `frame_pose`;
- `target`;
- `error`;
- `velocity`.

## Servicios

### `set_task_enabled`

```text
Servicio: /set_task_enabled
Tipo: task_priority_kinematic_control/srv/SetTaskEnabled
```

Request:

```text
string task_id
bool enabled
```

Activa o desactiva una tarea.

Ejemplo:

```bash
ros2 service call /set_task_enabled \
  task_priority_kinematic_control/srv/SetTaskEnabled \
  "{task_id: left_pose, enabled: false}"
```

### `set_task_disabled`

Solo en `TaskPriorityController`:

```text
Servicio: /set_task_disabled
Tipo: task_priority_kinematic_control/srv/SetTaskDisabled
```

Request:

```text
string task_id
```

Es equivalente a llamar `set_task_enabled` con `enabled: false`.

### `reorder_tasks`

```text
Servicio: /reorder_tasks
Tipo: task_priority_kinematic_control/srv/ReorderTasks
```

Request:

```text
string[] ordered_task_ids
```

Reordena todas las tareas. La lista debe contener exactamente todos los IDs actuales. El primer elemento pasa a prioridad `0`, el segundo a prioridad `1`, etc.

Ejemplo:

```bash
ros2 service call /reorder_tasks \
  task_priority_kinematic_control/srv/ReorderTasks \
  "{ordered_task_ids: [joint_limits, left_pose, right_pose, left_arm_nominal, right_arm_nominal, base_yaw]}"
```

### `list_tasks`

```text
Servicio: /list_tasks
Tipo: task_priority_kinematic_control/srv/ListTasks
```

No tiene campos de request. Devuelve:

- lista de tareas;
- backend;
- método de solver;
- `success`;
- `message`.

### `switch_backend`

Solo en `task_priority_runtime`:

```text
Servicio: /switch_backend
Tipo: task_priority_kinematic_control/srv/SwitchBackend
```

Request:

```text
string backend_name
```

Ejemplo:

```bash
ros2 service call /switch_backend \
  task_priority_kinematic_control/srv/SwitchBackend \
  "{backend_name: task_priority_kinematic_control/KDLKinematicsBackend}"
```

Nota: aunque existe el backend Pinocchio, actualmente no es operativo para ejecución.

### `reset_solver`

Solo en `task_priority_runtime`:

```text
Servicio: /reset_solver
Tipo: std_srvs/srv/Trigger
```

Llama a `reset()` en las tareas. En la implementación actual las tareas no guardan integradores internos, así que este reset es principalmente un punto de extensión.

## Mensajes

### `TaskStatus`

```text
string id
string plugin
string group
uint32 priority
bool enabled
bool active
string status_message
float64[] error
float64[] command
string target_type
string[] joint_names
```

Describe una tarea configurada. `target_type` puede ser `pose`, `joint_array` o `none`.

### `HierarchyState`

```text
std_msgs/Header header
string backend_name
string solver_method
bool ready
TaskStatus[] tasks
```

Estado global de la jerarquía.

### `TaskState`

```text
string id
string type
bool enabled
bool active
bool has_frame_pose
string frame_id
geometry_msgs/Pose frame_pose
float64[] target
float64[] feedforward
float64[] error
float64[] velocity
```

Estado numérico de una tarea. Actualmente lo publica el controlador `ros2_control`.

`frame_pose` contiene la pose FK actual del frame asociado cuando la tarea tiene frame físico claro:

- `left_pose` o `right_pose`: pose actual del efector;
- `base_pose` o `base_yaw`: pose actual de `base_frame`;
- tareas articulares como `joint_limits`: `has_frame_pose=false`.

### `ControllerOutput`

```text
std_msgs/Header header
string[] generalized_velocity_names
float64[] generalized_velocity
string[] base_velocity_names
float64[] base_velocity
string[] left_joint_names
float64[] left_arm_velocity
string[] right_joint_names
float64[] right_arm_velocity
```

Salida final del solver de `TaskPriorityController`. Permite ver en un único mensaje las velocidades de la base y de los dos brazos. La velocidad de base está expresada en `world_frame`; el comando enviado a `body_velocity` se convierte después a `base_frame`.

### `SolverDiagnostics`

```text
std_msgs/Header header
string backend_name
string solver_method
float64 damped_least_squares_lambda
float64 max_command_ratio
float64 max_joint_ratio
float64 condition_estimate
bool state_valid
string message
```

Diagnóstico del solver. En el código actual se rellenan backend, método, lambda, condición, validez y mensaje; los ratios quedan reservados para ampliaciones.

## Ejemplos de comandos

Enviar objetivo de pose al efector izquierdo:

```bash
ros2 topic pub --once /cirtesub/controller/task_priority/tasks/left_pose/target \
  geometry_msgs/msg/PoseStamped \
  "{header: {frame_id: world_ned}, pose: {position: {x: 0.5, y: 0.2, z: -0.1}, orientation: {w: 1.0}}}"
```

Cambiar postura nominal del brazo izquierdo:

```bash
ros2 topic pub --once /cirtesub/controller/task_priority/tasks/left_arm_nominal/joint_target \
  std_msgs/msg/Float64MultiArray \
  "{data: [0.0, 1.2, 1.2, 1.0, 0.0]}"
```

Listar tareas:

```bash
ros2 service call /list_tasks task_priority_kinematic_control/srv/ListTasks "{}"
```

Ver estado global:

```bash
ros2 topic echo /hierarchy_state
```

## Build y tests

Compilar desde el workspace:

```bash
colcon build --packages-select task_priority_kinematic_control
source install/setup.bash
```

Ejecutar tests del solver:

```bash
colcon test --packages-select task_priority_kinematic_control
colcon test-result --verbose
```

## Dependencias principales

El paquete depende de:

- `rclcpp`;
- `controller_interface`;
- `hardware_interface`;
- `pluginlib`;
- `realtime_tools`;
- `geometry_msgs`;
- `sensor_msgs`;
- `std_msgs`;
- `std_srvs`;
- `sura_msgs`;
- `kdl_parser`;
- `urdf`;
- `Eigen3`;
- `rosidl_default_generators`;
- `rosidl_default_runtime`.

## Archivos importantes

| Archivo | Función |
| --- | --- |
| `src/core/hierarchy_solver.cpp` | Solver jerárquico y saturación de velocidades. |
| `src/core/task_manager.cpp` | Carga, ordena y gestiona tareas con pluginlib. |
| `src/core/runtime_node.cpp` | Nodo standalone, topics, servicios y timer. |
| `src/core/task_priority_controller.cpp` | Plugin `ros2_control`. |
| `src/core/whole_body_model.cpp` | Frames, joints y offsets del vector generalizado. |
| `src/kinematics/kdl_kinematics_backend.cpp` | Backend KDL funcional. |
| `src/tasks/*.cpp` | Implementación de cada tarea. |
| `config/task_priority_kinematic_control.yaml` | Ejemplo para nodo runtime. |
| `config/task_priority_controller_ros2_control.yaml` | Ejemplo para `ros2_control`. |
| `task_priority_kinematic_control_core_plugins.xml` | Registro de backends y tasks. |
| `task_priority_kinematic_control_plugins.xml` | Registro del controlador `ros2_control`. |

## Notas de uso

- Los tamaños de `dof_weights` y `velocity_limits` deben coincidir con `total_dofs` para cubrir todos los DOFs.
- Los límites articulares se ordenan como `left_arm_joints` seguido de `right_arm_joints`.
- `priority` menor significa tarea más importante.
- Para que una task de pose se mueva necesita recibir un objetivo, salvo tasks con objetivo por defecto como `EndEffectorPositionTask`.
- El backend Pinocchio está registrado, pero no debe usarse en ejecución hasta que se implemente.
- Si el nodo no recibe navegación o joints, no publica comandos útiles y reporta `waiting_for_state`.

## Cápsulas manuales para evitar autocolisiones

`SelfCollisionAvoidanceTask` utiliza cápsulas definidas en parámetros YAML, sin extraerlas de mallas DAE. La [documentación de cápsulas manuales](docs/self_collision_avoidance.md) describe la configuración, las parejas explícitas, la parada por degeneración y la visualización RViz fuera del bucle de control.

## Feedforward de las tareas de pose

`FramePoseTask` y `EndEffectorPoseTask` reciben `geometry_msgs/msg/TwistStamped` en
`task_priority/tasks/<id>/feedforward`, junto al tópico de objetivo existente
`task_priority/tasks/<id>/target`. En el bringup de CIRTESUB, el prefijo público es
`/cirtesub/controller/task_priority/tasks`.

La velocidad deseada es `vff + K * error`, con orden XYZ lineal y XYZ angular.
`header.frame_id` selecciona los ejes de la velocidad; vacío utiliza el frame de
la base. La rotación hacia el mundo se resuelve por TF en el callback del mensaje,
sin desplazar el punto controlado ni introducir un término por cambio de origen.
Timestamp cero utiliza la transformación más reciente. Mensajes no finitos o
referencias no disponibles borran el feedforward. El feedforward no tiene timeout;
para detenerlo hay que publicar cero o deshabilitar/desactivar/reiniciar el control.

Cada tarea admite estas opciones de configuración, también actualizables mediante
parámetros ROS durante la ejecución:

```yaml
tasks.left_pose.auto_update_target_pose: true
tasks.left_pose.feedforward_velocity_threshold: 0.001
```

Con actualización automática, la tarea captura su pose al habilitarse y actualiza
el objetivo en cada ciclo con `norm(vff) > threshold`. Cuando la norma es menor o
igual, conserva el último objetivo. El umbral es la norma directa de los seis
componentes, sin ponderación entre velocidades lineales y angulares, y no recorta
el feedforward. El objetivo actualizado se publica también en `/target`; el
controlador ignora sus propias publicaciones para evitar realimentar objetivos
atrasados. `TaskState.feedforward` muestra la velocidad aplicada en el mundo.

Con `auto_update_target_pose: false`, se respeta el objetivo recibido por `/target`
y solamente se suma la velocidad. Las demás tareas no reciben estos tópicos.
