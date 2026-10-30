import blenderproc as bproc

# Bootstrap BlenderProc's embedded-Python dependencies once in the image layer.
# Upstream setup owns their pins; no GPU or rendering is required here.
bproc.init()
print("BlenderProc embedded Python is ready")
