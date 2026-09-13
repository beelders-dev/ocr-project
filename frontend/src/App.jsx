import { useState } from "react";
import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

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
            <label className="primary-button">
              <span className="button-icon">↑</span>
              Upload Receipt
              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
              />
            </label>

            <p className="supported-text">
              JPG, PNG or HEIC · Take a clear photo
            </p>
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
