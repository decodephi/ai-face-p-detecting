import { AnalysisResponse } from "../types/analysis";

interface AnalysisResultPanelProps {
  result: AnalysisResponse | null;
  isLoading: boolean;
  error: string | null;
  loadingMessage: string;
}

export function AnalysisResultPanel({
  result,
  isLoading,
  error,
  loadingMessage,
}: AnalysisResultPanelProps) {
  const annotatedImageSrc = result
    ? result.annotated_image.startsWith("data:")
      ? result.annotated_image
      : `data:image/png;base64,${result.annotated_image}`
    : null;

  const hasFindings = Boolean(result && (result.pimples.length > 0 || result.dark_spots_area > 0.01));

  return (
    <section className="panel results-panel">
      <div className="panel-copy">
        <p className="eyebrow">Analysis Output</p>
        <h2>Annotated image and analysis report.</h2>
      </div>

      {isLoading && (
        <div className="status-card scanning-card">
          <div className="scan-spinner" />
          <div>
            <strong>Scanning image</strong>
            <p>{loadingMessage}</p>
          </div>
        </div>
      )}
      {error && <div className="error-card">{error}</div>}

      {!isLoading && !error && !result && (
        <div className="status-card">Submit an image to receive detections, oiliness classification, and overlays.</div>
      )}

      {result && (
        <div className="results-stack">
          <div className="results-top-row">
            <div className="annotated-panel">
              <div className="annotated-card annotated-card-full">
                {annotatedImageSrc ? (
                  <img alt="Annotated skin analysis" className="annotated-image" src={annotatedImageSrc} />
                ) : (
                  <div className="empty-preview">Annotated output is unavailable for this response.</div>
                )}
              </div>

              <div className="legend-row">
                <span><i className="legend-dot pimple-dot" /> Pimples</span>
                <span><i className="legend-dot whitehead-dot" /> Whiteheads</span>
                <span><i className="legend-dot blackhead-dot" /> Blackheads</span>
                <span><i className="legend-dot darkspot-dot" /> Dark spots</span>
                <span><i className="legend-dot oil-dot" /> Oily regions</span>
              </div>
            </div>

          </div>

          <div className="summary-card report-card overview-card">
            <h3>Overview</h3>
            <div className="metric-row">
              <span>Skin type</span>
              <strong>{result.oiliness.type}</strong>
            </div>
            <div className="metric-row">
              <span>Oiliness score</span>
              <strong>{result.oiliness.score.toFixed(2)}</strong>
            </div>
            <div className="metric-row">
              <span>Dark spot area</span>
              <strong>{(result.dark_spots_area * 100).toFixed(2)}%</strong>
            </div>
            <div className="metric-row">
              <span>Total visible lesions</span>
              <strong>{result.pimples.length}</strong>
            </div>
          </div>

          <div className="report-grid report-grid-below">
            <div className="summary-card report-card report-card-wide interpretation-card">
              <h3>Interpretation</h3>
              <p className="report-copy">
                {hasFindings
                  ? "Highlighted regions indicate the strongest visible signs detected by the current image-analysis pipeline. Use the colored boxes and contours on the image to inspect exact locations."
                  : "The current image did not produce strong visible lesion detections. Better lighting and a sharper close-up image will usually improve the result."}
              </p>
              <p className="report-copy">
                Oiliness is estimated from shine and smoothness patterns, while dark spots are mapped from low-lightness skin regions relative to surrounding skin.
              </p>
            </div>
          </div>
        </div>
      )}
    </section>
  );
}
