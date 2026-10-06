# Examples set aside

They were moved here so that `examples/` holds **one single example**: the square mission, in three
files that are the whole argument (one behaviour, two scenarios). These still work, but they are not
the front door.

| file | what for |
|---|---|
| `demo_para_depurar.py` | F5 in VS Code and it runs. It carries the runner distilled to 30 lines (`MiniRunner`): the best place to put breakpoints and watch what the host does to the protocol. |
| `ejemplo_colega_vuelo_y_vision.py` | **to read, not to run** (it uses `OnboardCamera`, which needs `picamera2` and only exists on the Raspberry). |
| `ejemplo_mision_100m.py` | likewise: to read. |
