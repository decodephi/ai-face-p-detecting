/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_API_BASE_URL?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}

declare class FaceDetector {
  constructor(options?: { fastMode?: boolean; maxDetectedFaces?: number });
  detect(input: CanvasImageSource): Promise<Array<{ boundingBox: DOMRectReadOnly }>>;
}

interface Window {
  FaceDetector?: typeof FaceDetector;
}
