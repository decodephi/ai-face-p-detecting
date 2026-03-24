import { ChangeEvent, RefObject } from "react";

type CaptureGuidance = {
  lightingGood: boolean;
  lookingStraight: boolean;
  positionGood: boolean;
  message: string;
};

interface ImageSourcePanelProps {
  isCameraOpen: boolean;
  imageUrl: string | null;
  countdown: number | null;
  captureGuidance: CaptureGuidance;
  fileInputRef: RefObject<HTMLInputElement | null>;
  videoRef: RefObject<HTMLVideoElement | null>;
  onFileSelect: (event: ChangeEvent<HTMLInputElement>) => void;
  onOpenCamera: () => void;
  onCaptureFrame: () => void;
  onCloseCamera: () => void;
  onRemoveImage: () => void;
}

export function ImageSourcePanel({
  isCameraOpen,
  imageUrl,
  countdown,
  captureGuidance,
  fileInputRef,
  videoRef,
  onFileSelect,
  onOpenCamera,
  onCaptureFrame,
  onCloseCamera,
  onRemoveImage,
}: ImageSourcePanelProps) {
  return (
    <section className="panel source-panel">
      <div className="panel-copy">
        <p className="eyebrow">Image Input</p>
        <h2>Upload a portrait or capture one live.</h2>
        <p className="muted">
          The pipeline expects a clear frontal face image. Preview updates before analysis.
        </p>
      </div>

      <div className="controls">
        <label className="upload-button">
          <span>{imageUrl ? "Replace Image" : "Upload Image"}</span>
          <input ref={fileInputRef} accept="image/*" type="file" onChange={onFileSelect} />
        </label>

        {!isCameraOpen ? (
          <button className="secondary-button" onClick={onOpenCamera} type="button">
            Open Webcam
          </button>
        ) : (
          <div className="camera-actions">
            <button className="secondary-button" onClick={onCaptureFrame} type="button">
              Capture Frame
            </button>
            <button className="ghost-button" onClick={onCloseCamera} type="button">
              Close Camera
            </button>
          </div>
        )}

        {imageUrl && !isCameraOpen && (
          <button className="ghost-button" onClick={onRemoveImage} type="button">
            Remove Image
          </button>
        )}
      </div>

      <div className="preview-card">
        {isCameraOpen ? (
          <div className="camera-shell">
            <video ref={videoRef} autoPlay muted playsInline className="preview-media" />
            <div className="camera-guidance">
              <div className="guide-badges">
                <span className={captureGuidance.lightingGood ? "guide-pill good" : "guide-pill"}>
                  Lighting
                </span>
                <span className={captureGuidance.lookingStraight ? "guide-pill good" : "guide-pill"}>
                  Look Straight
                </span>
                <span className={captureGuidance.positionGood ? "guide-pill good" : "guide-pill"}>
                  Position
                </span>
              </div>

              <div className="face-frame" />
              <div className="guide-message">{captureGuidance.message}</div>
              <div className="guide-notes">
                <span>Use bright front lighting</span>
                <span>Keep your face inside the oval</span>
                <span>Hold still and look straight</span>
              </div>
              {countdown !== null && <div className="countdown-badge">{countdown}</div>}
            </div>
          </div>
        ) : imageUrl ? (
          <img src={imageUrl} alt="Selected preview" className="preview-media" />
        ) : (
          <div className="empty-preview">No image selected yet.</div>
        )}
      </div>
    </section>
  );
}
