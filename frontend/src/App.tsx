import { ChangeEvent, useEffect, useRef, useState } from "react";

import { AnalysisResultPanel } from "./components/AnalysisResultPanel";
import { ImageSourcePanel } from "./components/ImageSourcePanel";
import { AnalysisResponse } from "./types/analysis";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL ?? (import.meta.env.DEV ? "http://127.0.0.1:8000" : "/api");
const API_URL = `${API_BASE_URL}/analyze`;
const SCAN_STEPS = [
  "Scanning skin texture...",
  "Mapping dark spots and visible lesions...",
  "Estimating oiliness and preparing report...",
] as const;

type CaptureGuidance = {
  lightingGood: boolean;
  lookingStraight: boolean;
  positionGood: boolean;
  message: string;
};

type EstimatedFaceBox = {
  x: number;
  y: number;
  width: number;
  height: number;
};

function dataUrlToFile(dataUrl: string, filename: string): File {
  const [metadata, base64Content] = dataUrl.split(",");
  const mimeMatch = metadata.match(/data:(.*?);base64/);
  const mimeType = mimeMatch ? mimeMatch[1] : "image/png";
  const bytes = atob(base64Content);
  const byteNumbers = new Array(bytes.length);

  for (let index = 0; index < bytes.length; index += 1) {
    byteNumbers[index] = bytes.charCodeAt(index);
  }

  return new File([new Uint8Array(byteNumbers)], filename, { type: mimeType });
}

function estimateSkinFaceBox(
  data: Uint8ClampedArray,
  width: number,
  height: number,
): EstimatedFaceBox | null {
  let minX = width;
  let minY = height;
  let maxX = 0;
  let maxY = 0;
  let skinPixels = 0;

  for (let index = 0; index < data.length; index += 4) {
    const r = data[index];
    const g = data[index + 1];
    const b = data[index + 2];
    const max = Math.max(r, g, b);
    const min = Math.min(r, g, b);
    const x = (index / 4) % width;
    const y = Math.floor(index / 4 / width);

    const isSkin =
      r > 45 &&
      g > 34 &&
      b > 20 &&
      max - min > 12 &&
      Math.abs(r - g) > 8 &&
      r > g &&
      r > b;

    if (!isSkin) {
      continue;
    }

    skinPixels += 1;
    minX = Math.min(minX, x);
    minY = Math.min(minY, y);
    maxX = Math.max(maxX, x);
    maxY = Math.max(maxY, y);
  }

  if (skinPixels < width * height * 0.07) {
    return null;
  }

  return {
    x: minX,
    y: minY,
    width: Math.max(1, maxX - minX),
    height: Math.max(1, maxY - minY),
  };
}

export default function App() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isCameraOpen, setIsCameraOpen] = useState(false);
  const [cameraStream, setCameraStream] = useState<MediaStream | null>(null);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [scanStepIndex, setScanStepIndex] = useState(0);
  const [captureGuidance, setCaptureGuidance] = useState<CaptureGuidance>({
    lightingGood: false,
    lookingStraight: false,
    positionGood: false,
    message: "Open the webcam and center your face in the guide.",
  });
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const countdownStartedRef = useRef(false);
  const stableFramesRef = useRef(0);
  const faceDetectorRef = useRef<FaceDetector | null>(null);

  useEffect(() => {
    if ("FaceDetector" in window) {
      faceDetectorRef.current = new FaceDetector({
        fastMode: true,
        maxDetectedFaces: 1,
      });
    }
  }, []);

  useEffect(() => {
    return () => {
      if (imageUrl?.startsWith("blob:")) {
        URL.revokeObjectURL(imageUrl);
      }
      streamRef.current?.getTracks().forEach((track) => track.stop());
    };
  }, [imageUrl]);

  useEffect(() => {
    if (!isCameraOpen || !cameraStream || !videoRef.current) {
      return;
    }

    const videoElement = videoRef.current;
    videoElement.srcObject = cameraStream;
    void videoElement.play().catch(() => {
      setError("Webcam preview could not start. Please retry opening the camera.");
    });
  }, [cameraStream, isCameraOpen]);

  useEffect(() => {
    if (!isCameraOpen || !videoRef.current) {
      return;
    }

    let cancelled = false;
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d");
    const sampleCanvas = document.createElement("canvas");
    const sampleContext = sampleCanvas.getContext("2d");

    const evaluateFrame = async () => {
      if (cancelled || !videoRef.current || !context || !sampleContext) {
        return;
      }

      const video = videoRef.current;
      if (!video.videoWidth || !video.videoHeight) {
        return;
      }

      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      context.drawImage(video, 0, 0, canvas.width, canvas.height);

      const sampleWidth = 160;
      const sampleHeight = Math.max(120, Math.round((canvas.height / canvas.width) * sampleWidth));
      sampleCanvas.width = sampleWidth;
      sampleCanvas.height = sampleHeight;
      sampleContext.drawImage(canvas, 0, 0, sampleWidth, sampleHeight);

      const imageData = sampleContext.getImageData(0, 0, sampleWidth, sampleHeight).data;
      let totalBrightness = 0;
      let totalDeviation = 0;

      for (let index = 0; index < imageData.length; index += 4) {
        totalBrightness += (imageData[index] + imageData[index + 1] + imageData[index + 2]) / 3;
      }

      const meanBrightness = totalBrightness / (imageData.length / 4);

      for (let index = 0; index < imageData.length; index += 4) {
        const brightness = (imageData[index] + imageData[index + 1] + imageData[index + 2]) / 3;
        totalDeviation += Math.abs(brightness - meanBrightness);
      }

      const meanDeviation = totalDeviation / (imageData.length / 4);
      const lightingGood = meanBrightness >= 105 && meanBrightness <= 210 && meanDeviation >= 18;
      let estimatedFaceBox: EstimatedFaceBox | null = null;

      if (faceDetectorRef.current) {
        try {
          const faces = await faceDetectorRef.current.detect(sampleCanvas);
          const face = faces[0];
          if (face?.boundingBox) {
            estimatedFaceBox = {
              x: face.boundingBox.x,
              y: face.boundingBox.y,
              width: face.boundingBox.width,
              height: face.boundingBox.height,
            };
          }
        } catch {
          estimatedFaceBox = estimateSkinFaceBox(imageData, sampleWidth, sampleHeight);
        }
      } else {
        estimatedFaceBox = estimateSkinFaceBox(imageData, sampleWidth, sampleHeight);
      }

      let lookingStraight = false;
      let positionGood = false;
      let message = "Bring your face into the frame.";

      if (estimatedFaceBox) {
        const { x, y, width, height } = estimatedFaceBox;
        const centerX = x + width / 2;
        const centerY = y + height / 2;
        const widthRatio = width / sampleWidth;
        const heightRatio = height / sampleHeight;

        lookingStraight =
          Math.abs(centerX - sampleWidth / 2) < sampleWidth * 0.09 &&
          Math.abs(centerY - sampleHeight * 0.5) < sampleHeight * 0.1;
        positionGood =
          widthRatio > 0.24 &&
          widthRatio < 0.52 &&
          heightRatio > 0.34 &&
          heightRatio < 0.7 &&
          y > sampleHeight * 0.08 &&
          y < sampleHeight * 0.28;

        if (!lightingGood) {
          message = "Improve your lighting and avoid dark shadows.";
        } else if (!lookingStraight) {
          message = "Look straight into the camera.";
        } else if (!positionGood) {
          message = "Move a little closer and keep your face centered.";
        } else {
          message = "Hold still. Auto capture will start in a moment.";
        }
      } else if (!lightingGood) {
        message = "Improve your lighting and avoid dark shadows.";
      }

      const frameReady = lightingGood && lookingStraight && positionGood && Boolean(estimatedFaceBox);
      stableFramesRef.current = frameReady ? stableFramesRef.current + 1 : 0;

      setCaptureGuidance({
        lightingGood,
        lookingStraight,
        positionGood,
        message,
      });
    };

    const intervalId = window.setInterval(() => {
      void evaluateFrame();
    }, 500);

    return () => {
      cancelled = true;
      stableFramesRef.current = 0;
      window.clearInterval(intervalId);
    };
  }, [isCameraOpen]);

  useEffect(() => {
    const allGood =
      isCameraOpen &&
      captureGuidance.lightingGood &&
      captureGuidance.lookingStraight &&
      captureGuidance.positionGood &&
      stableFramesRef.current >= 4;

    if (!allGood) {
      countdownStartedRef.current = false;
      setCountdown(null);
      return;
    }

    if (countdownStartedRef.current) {
      return;
    }

    countdownStartedRef.current = true;
    setCountdown(3);

    const timerId = window.setInterval(() => {
      const stillReady =
        captureGuidance.lightingGood &&
        captureGuidance.lookingStraight &&
        captureGuidance.positionGood &&
        stableFramesRef.current >= 4;

      if (!stillReady) {
        window.clearInterval(timerId);
        countdownStartedRef.current = false;
        setCountdown(null);
        return;
      }

      setCountdown((current) => {
        if (current === null) {
          return null;
        }
        if (current <= 1) {
          window.clearInterval(timerId);
          captureFrame();
          return null;
        }
        return current - 1;
      });
    }, 1000);

    return () => {
      window.clearInterval(timerId);
    };
  }, [captureGuidance, isCameraOpen]);

  useEffect(() => {
    if (!isLoading) {
      setScanStepIndex(0);
      return;
    }

    const intervalId = window.setInterval(() => {
      setScanStepIndex((current) => (current + 1) % SCAN_STEPS.length);
    }, 1300);

    return () => {
      window.clearInterval(intervalId);
    };
  }, [isLoading]);

  const clearSelectedImage = () => {
    if (imageUrl?.startsWith("blob:")) {
      URL.revokeObjectURL(imageUrl);
    }

    setSelectedFile(null);
    setImageUrl(null);
    setResult(null);
    setError(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleFileSelect = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) {
      return;
    }

    if (imageUrl?.startsWith("blob:")) {
      URL.revokeObjectURL(imageUrl);
    }

    setSelectedFile(file);
    setImageUrl(URL.createObjectURL(file));
    setResult(null);
    setError(null);
    closeCamera();
  };

  const openCamera = async () => {
    setError(null);
    setResult(null);
    setCountdown(null);
    countdownStartedRef.current = false;
    stableFramesRef.current = 0;

    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user" } });
      streamRef.current = stream;
      setCameraStream(stream);
      setIsCameraOpen(true);
    } catch {
      setError("Unable to access the webcam. Please verify browser permissions.");
    }
  };

  const closeCamera = () => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraStream(null);
    setIsCameraOpen(false);
    setCountdown(null);
    countdownStartedRef.current = false;
    stableFramesRef.current = 0;
    setCaptureGuidance({
      lightingGood: false,
      lookingStraight: false,
      positionGood: false,
      message: "Open the webcam and center your face in the guide.",
    });

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
  };

  const captureFrame = () => {
    if (!videoRef.current || !videoRef.current.videoWidth || !videoRef.current.videoHeight) {
      setError("Webcam preview is not ready yet. Wait a moment and try capture again.");
      return;
    }

    const canvas = document.createElement("canvas");
    canvas.width = videoRef.current.videoWidth;
    canvas.height = videoRef.current.videoHeight;
    const context = canvas.getContext("2d");

    if (!context) {
      setError("Canvas capture is unavailable in this browser.");
      return;
    }

    context.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL("image/png");
    const capturedFile = dataUrlToFile(dataUrl, "webcam-capture.png");

    if (imageUrl?.startsWith("blob:")) {
      URL.revokeObjectURL(imageUrl);
    }

    setSelectedFile(capturedFile);
    setImageUrl(dataUrl);
    setResult(null);
    setError(null);

    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }

    closeCamera();
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError("Select or capture an image before starting analysis.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", selectedFile);

      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        throw new Error("Analysis request failed.");
      }

      const payload = (await response.json()) as AnalysisResponse;
      setResult(payload);
    } catch {
      setError("Analysis could not be completed. Confirm the FastAPI server is running.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">AI Facial Skin Analysis</p>
        <h1>Inspect facial skin conditions from a browser-based workflow.</h1>
        <p className="hero-copy">
          Upload a portrait or capture one live, then run a modular computer vision pipeline that highlights
          pimples, whiteheads, blackheads, dark spots, and oily regions.
        </p>
        <button className="primary-button" onClick={handleAnalyze} type="button">
          Analyze Skin Image
        </button>
      </section>

      <div className="workspace-grid">
        <ImageSourcePanel
          isCameraOpen={isCameraOpen}
          imageUrl={imageUrl}
          countdown={countdown}
          captureGuidance={captureGuidance}
          fileInputRef={fileInputRef}
          videoRef={videoRef}
          onFileSelect={handleFileSelect}
          onOpenCamera={openCamera}
          onCaptureFrame={captureFrame}
          onCloseCamera={closeCamera}
          onRemoveImage={clearSelectedImage}
        />
        <AnalysisResultPanel
          result={result}
          isLoading={isLoading}
          error={error}
          loadingMessage={SCAN_STEPS[scanStepIndex]}
        />
      </div>
    </main>
  );
}
