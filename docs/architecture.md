# Architecture

## Overview

The system is split into:

- `frontend`: React + TypeScript client
- `backend`: FastAPI service with modular CV pipeline

## Extensibility

The lesion detector exposes a single interface returning labeled bounding boxes. Replacing the heuristic engine with YOLO only requires swapping detector internals while preserving the response schema.
