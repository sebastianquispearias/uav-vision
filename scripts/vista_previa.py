"""Live MJPEG preview of the ArduCam, to aim the camera before a test.

Serves http://<pi>:8000/ with the same capture transform the deployment chain
uses (rot180), so what you see is what the detector gets. Stop it with Ctrl+C
or `pkill -f vista_previa` before running anything else that needs the camera.
"""

import io
import logging
from http import server
from threading import Condition

from libcamera import Transform
from picamera2 import Picamera2
from picamera2.encoders import MJPEGEncoder
from picamera2.outputs import FileOutput

PAGINA = """\
<html><head><title>ArduCam</title></head>
<body style="margin:0;background:#000;text-align:center">
<img src="stream.mjpg" style="max-width:100%;height:auto"/>
</body></html>
"""


class Salida(io.BufferedIOBase):
    def __init__(self):
        self.frame = None
        self.condition = Condition()

    def write(self, buf):
        with self.condition:
            self.frame = buf
            self.condition.notify_all()


class Handler(server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            cuerpo = PAGINA.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(cuerpo)))
            self.end_headers()
            self.wfile.write(cuerpo)
        elif self.path == "/stream.mjpg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Content-Type",
                             "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()
            try:
                while True:
                    with salida.condition:
                        salida.condition.wait()
                        frame = salida.frame
                    self.wfile.write(b"--FRAME\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(frame)))
                    self.end_headers()
                    self.wfile.write(frame)
                    self.wfile.write(b"\r\n")
            except Exception:
                pass  # client closed the tab
        else:
            self.send_error(404)


picam = Picamera2()
picam.configure(picam.create_video_configuration(
    main={"size": (960, 540)}, transform=Transform(hflip=1, vflip=1)))
salida = Salida()
picam.start_recording(MJPEGEncoder(), FileOutput(salida))
try:
    logging.info("preview en http://0.0.0.0:8000/")
    server.ThreadingHTTPServer(("", 8000), Handler).serve_forever()
finally:
    picam.stop_recording()
