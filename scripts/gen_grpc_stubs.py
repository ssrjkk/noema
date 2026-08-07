#!/usr/bin/env python3
"""Regenerate gRPC stubs for protos/noema_engine.proto into noema/grpc/.

Usage: python scripts/gen_grpc_stubs.py
"""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTO = os.path.join(ROOT, "protos", "noema_engine.proto")
OUT = os.path.join(ROOT, "noema", "grpc")
PB2_GRPC = os.path.join(OUT, "noema_engine_pb2_grpc.py")

cmd = [
    sys.executable,
    "-m",
    "grpc_tools.protoc",
    "-I",
    os.path.join(ROOT, "protos"),
    "--python_out=" + OUT,
    "--grpc_python_out=" + OUT,
    "--pyi_out=" + OUT,
    PROTO,
]
subprocess.run(cmd, check=True)

# grpc_python_plugin emits a bare "import noema_engine_pb2" for protos at the
# include root; qualify it so it resolves as noema.grpc.noema_engine_pb2.
with open(PB2_GRPC, encoding="utf-8") as fh:
    src = fh.read()
src = src.replace(
    "import noema_engine_pb2 as noema__engine__pb2",
    "from noema.grpc import noema_engine_pb2 as noema__engine__pb2",
)
with open(PB2_GRPC, "w", encoding="utf-8") as fh:
    fh.write(src)
print("Generated:", os.path.relpath(PB2_GRPC, ROOT))
