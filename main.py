# custom_response_server.py
from http.server import BaseHTTPRequestHandler, HTTPServer

class MyHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()
        self.wfile.write(b"<h1>Hello from Python Server!</h1>")

server = HTTPServer(('localhost', 8080), MyHandler)
print("Server started on http://localhost:8080")
server.serve_forever()
