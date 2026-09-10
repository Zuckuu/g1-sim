"""Empty GUI startup test; no robot control or external robot communication."""

from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": False,
        "width": 640,
        "height": 480,
        "window_width": 960,
        "window_height": 640,
        "renderer": "RayTracedLighting",
        "anti_aliasing": 0,
        "multi_gpu": False,
    }
)

try:
    for _ in range(120):
        if not app.is_running():
            raise RuntimeError("Simulator closed before completing the startup test")
        app.update()
    print("ISAAC_SMOKE_OK: Application initialized and completed 120 update frames.", flush=True)
finally:
    app.close()
