import { useEffect, useRef, useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [cameraOpen, setCameraOpen] = useState(false);
  const [cameraError, setCameraError] = useState(null);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);

  async function openCamera() {
    setCameraError(null);
    setCameraOpen(true);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: {
          facingMode: {
            ideal: "environment",
          },
        },
        audio: false,
      });

      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (error) {
      console.error("Camera error:", error);
      setCameraError(
        "Unable to access the camera. Please check your browser permission.",
      );
    }
  }

  function closeCamera() {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }

    setCameraOpen(false);
    setCameraError(null);
  }

  function capturePhoto() {
    const video = videoRef.current;
    const canvas = canvasRef.current;

    if (!video || !canvas) return;

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(
      (blob) => {
        if (!blob) return;

        const file = new File([blob], `receipt-${Date.now()}.jpg`, {
          type: "image/jpeg",
        });

        setImage(file);
        setResult(null);
        setError(null);

        closeCamera();
      },
      "image/jpeg",
      0.92,
    );
  }

  function handleImageChange(event) {
    const file = event.target.files[0];

    if (!file) return;

    setImage(file);
    setResult(null);
    setError(null);
  }

  async function processReceipt() {
    if (!image) return;

    setLoading(true);
    setError(null);
    setResult(null);

    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/api/receipts/process/",
        {
          method: "POST",
          body: formData,
        },
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.detail || "Failed to process receipt.");
      }

      setResult(data.data);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  }

  function resetReceipt() {
    setImage(null);
    setResult(null);
    setError(null);
  }

  return (
    <main className="app">
      {cameraOpen && (
        <div className="camera-overlay">
          <div className="camera-header">
            <button className="camera-close" onClick={closeCamera}>
              ×
            </button>

            <span>Scan Receipt</span>

            <div className="camera-spacer"></div>
          </div>

          <div className="camera-view">
            <video ref={videoRef} autoPlay playsInline muted />

            <div className="camera-frame">
              <span className="camera-corner top-left"></span>
              <span className="camera-corner top-right"></span>
              <span className="camera-corner bottom-left"></span>
              <span className="camera-corner bottom-right"></span>
            </div>

            <p className="camera-hint">Position the receipt inside the frame</p>
          </div>

          {cameraError ? (
            <div className="camera-error">{cameraError}</div>
          ) : (
            <button className="capture-button" onClick={capturePhoto}>
              <span></span>
            </button>
          )}

          <p className="camera-action-label">
            {cameraError ? "Camera unavailable" : "Capture"}
          </p>

          <canvas ref={canvasRef} className="camera-canvas" />
        </div>
      )}
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">R</div>
          <span>Receiptly</span>
        </div>

        <span className="status-dot">
          <span></span>
          OCR Ready
        </span>
      </header>

      <section className="hero">
        <p className="hero-label">RECEIPT SCANNER</p>

        <h1>
          Turn your receipt
          <br />
          into <span>data.</span>
        </h1>

        <p className="hero-description">
          Upload a receipt and automatically extract the important details.
        </p>
      </section>

      {!image && (
        <section className="scanner-area">
          <div className="receipt-placeholder">
            <div className="receipt-paper">
              <div className="receipt-logo"></div>
              <div className="receipt-line large"></div>
              <div className="receipt-line"></div>
              <div className="receipt-line short"></div>

              <div className="receipt-items">
                <span></span>
                <span></span>
                <span></span>
                <span></span>
              </div>

              <div className="receipt-total">
                <span>TOTAL</span>
                <strong>₱170.00</strong>
              </div>
            </div>

            <div className="scan-corners">
              <i className="corner top-left"></i>
              <i className="corner top-right"></i>
              <i className="corner bottom-left"></i>
              <i className="corner bottom-right"></i>
            </div>
          </div>

          <div className="scanner-actions">
            <button className="primary-button" onClick={openCamera}>
              <span className="button-icon">⌾</span>
              <span>Scan Receipt</span>
            </button>

            <label className="gallery-button">
              Upload from Gallery
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
              />
            </label>

            <p className="supported-text">JPG, PNG or HEIC</p>
          </div>
        </section>
      )}

      {image && (
        <section className="receipt-section">
          <div className="receipt-header">
            <div>
              <p className="section-label">YOUR RECEIPT</p>
              <h2>Ready to scan</h2>
            </div>

            <button className="reset-button" onClick={resetReceipt}>
              Change
            </button>
          </div>

          <div className="receipt-preview">
            <img src={URL.createObjectURL(image)} alt="Selected receipt" />

            {loading && (
              <div className="scanning-overlay">
                <div className="scan-line"></div>

                <div className="scanning-message">
                  <div className="spinner"></div>
                  <strong>Reading receipt...</strong>
                  <span>Extracting information</span>
                </div>
              </div>
            )}
          </div>

          {!loading && !result && (
            <button className="extract-button" onClick={processReceipt}>
              <span>Extract Information</span>
              <span className="arrow">→</span>
            </button>
          )}
        </section>
      )}

      {error && (
        <section className="error-message">
          <strong>We couldn't process that receipt.</strong>
          <span>{error}</span>
        </section>
      )}

      {result && (
        <section className="results-section">
          <div className="results-header">
            <div>
              <p className="section-label">EXTRACTION COMPLETE</p>
              <h2>Receipt details</h2>
            </div>

            <div className="success-icon">✓</div>
          </div>

          <div className="result-list">
            <ResultRow label="Company" value={result.company} />
            <ResultRow label="Date" value={result.date} />
            <ResultRow label="TIN" value={result.tin} />
            <ResultRow label="Invoice Number" value={result.invoice_number} />
            <ResultRow
              label="VATable Sales"
              value={
                result.vatable_sales !== null
                  ? `₱${Number(result.vatable_sales).toFixed(2)}`
                  : null
              }
            />
            <ResultRow
              label="VAT Amount"
              value={
                result.vat_amount !== null
                  ? `₱${Number(result.vat_amount).toFixed(2)}`
                  : null
              }
            />
          </div>

          <div className="total-row">
            <span>Total</span>
            <strong>
              {result.total !== null
                ? `₱${Number(result.total).toFixed(2)}`
                : "Not detected"}
            </strong>
          </div>

          <div
            className={`vat-result ${result.vat_valid ? "valid" : "review"}`}
          >
            <span>{result.vat_valid ? "✓" : "!"}</span>

            <div>
              <strong>
                {result.vat_valid
                  ? "VAT calculation verified"
                  : "VAT calculation needs review"}
              </strong>

              <small>
                {result.vat_valid
                  ? "The extracted VAT matches the expected calculation."
                  : "Please verify the extracted values against the receipt."}
              </small>
            </div>
          </div>

          <button className="scan-again-button" onClick={resetReceipt}>
            Scan another receipt
          </button>
        </section>
      )}

      <footer>
        <span>Powered by</span>
        <strong>PaddleOCR + LayoutLMv3</strong>
      </footer>
    </main>
  );
}

function ResultRow({ label, value }) {
  return (
    <div className="result-row">
      <span>{label}</span>
      <strong>{value || "Not detected"}</strong>
    </div>
  );
}

export default App;
