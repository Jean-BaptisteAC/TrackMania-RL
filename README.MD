<h2>Implemented</h2>

    - ...

<h2>TODO</h2>

    - Remove the thread architecture
    - Implement the detection of checkpoints and finish
    - Reset state of the game at every end of episode
    - To restart race, replace calls to keyboard input manager by TMInterface calls. Two potential options:
        - call respawn method: https://github.com/donadigo/TMInterfaceClientPython/blob/20a393b7ea28e71f0ef1270b6a94926778a6b3db/tminterface/interface.py#L340
        - save game state and restore it with rewind_to_state https://github.com/donadigo/TMInterfaceClientPython/blob/20a393b7ea28e71f0ef1270b6a94926778a6b3db/tminterface/interface.py#L453
    - Handle multiple instances of TMNF to gather data faster