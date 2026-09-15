import { useEffect, useRef, useState } from "react";
import { useReceiptScanner } from "./useReceiptScanner";
import * as XLSX from "xlsx";

import "./App.css";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [previewError, setPreviewError] = useState(false);
  const { scanImage, isProcessing } = useReceiptScanner();

  const [previewUrl, setPreviewUrl] = useState(null);

  const nativeCameraInputRef = useRef(null);

  useEffect(() => {
    if (!image) {
      setPreviewUrl(null);
      return;
    }

    const url = URL.createObjectURL(image);
    setPreviewUrl(url);

    return () => {
      URL.revokeObjectURL(url);
    };
  }, [image]);

  function handleImageChange(event) {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    if (!selectedFile.type.startsWith("image/")) {
      setImage(null);
      setResult(null);
      setError("Please upload a valid receipt image.");
      setPreviewError(false);
      return;
    }

    setError(null);
    setResult(null);
    setPreviewError(false);
    setImage(selectedFile);
  }

  async function handleNativeCameraChange(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    event.target.value = "";

    if (!file.type.startsWith("image/")) {
      setError("Please capture a valid receipt image.");
      return;
    }

    try {
      const dataUrl = await scanImage(file);

      const response = await fetch(dataUrl);
      const blob = await response.blob();

      const scannedFile = new File(
        [blob],
        `receipt-scanned-${Date.now()}.jpg`,
        {
          type: "image/jpeg",
        },
      );

      setImage(scannedFile);
      setResult(null);
      setError(null);
      setPreviewError(false);
    } catch (error) {
      setImage(null);
      setResult(null);
      setError(`Scanner error: ${error.message}`);
      setPreviewError(false);
    }
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
        `${import.meta.env.VITE_API_URL}/api/receipts/process/`,
        {
          method: "POST",
          body: formData,
        },
      );

      const data = await response.json();

      if (!response.ok) {
        const errorMessage =
          data.image?.[0] ||
          data.error ||
          data.detail ||
          "Failed to process receipt.";

        throw new Error(errorMessage);
      }

      setResult(data.data);
    } catch (error) {
      setError(error.message);
    } finally {
      setLoading(false);
    }
  }

  function downloadExcel() {
    if (!result) return;

    const exportedAt = new Date();

    const data = [
      ["Receiptly — Receipt Data"],
      [
        `Exported: ${exportedAt.toLocaleDateString()} ${exportedAt.toLocaleTimeString()}`,
      ],
      [],
      [
        "Company",
        "Date",
        "TIN",
        "Invoice Number",
        "VATable Sales",
        "VAT Amount",
        "Total",
        "VAT Valid",
      ],
      [
        result.company || "Not detected",
        result.date || "Not detected",
        result.tin || "Not detected",
        result.invoice_number || "Not detected",
        result.vatable_sales ?? "",
        result.vat_amount ?? "",
        result.total ?? "",
        result.vat_valid ? "Yes" : "Needs review",
      ],
    ];

    const worksheet = XLSX.utils.aoa_to_sheet(data);

    worksheet["!cols"] = [
      { wch: 28 },
      { wch: 18 },
      { wch: 18 },
      { wch: 20 },
      { wch: 18 },
      { wch: 16 },
      { wch: 16 },
      { wch: 16 },
    ];

    worksheet["!freeze"] = { xSplit: 0, ySplit: 4 };

    ["E5", "F5", "G5"].forEach((cell) => {
      if (worksheet[cell]) {
        worksheet[cell].z = "₱#,##0.00";
      }
    });

    worksheet["A1"].s = {
      font: {
        bold: true,
        sz: 16,
      },
    };

    const headerCells = ["A4", "B4", "C4", "D4", "E4", "F4", "G4", "H4"];

    headerCells.forEach((cell) => {
      worksheet[cell].s = {
        font: {
          bold: true,
        },
      };
    });

    const workbook = XLSX.utils.book_new();

    XLSX.utils.book_append_sheet(workbook, worksheet, "Receipt");

    const datePart = result.date
      ? String(result.date).replace(/[^\d-]/g, "-")
      : exportedAt.toISOString().slice(0, 10);

    XLSX.writeFile(workbook, `receiptly-${datePart}.xlsx`);
  }

  function resetReceipt() {
    setImage(null);
    setResult(null);
    setError(null);
    setPreviewError(false);
  }

  return (
    <main className="app">
      <input
        ref={nativeCameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleNativeCameraChange}
        hidden
      />

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
            <button
              className="primary-button"
              onClick={() => nativeCameraInputRef.current?.click()}
            >
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
            {!previewError ? (
              <img
                src={previewUrl}
                alt="Selected receipt"
                onError={() => {
                  setPreviewError(true);
                  setError("The selected image could not be previewed.");
                }}
              />
            ) : (
              <div className="receipt-preview-placeholder">
                <div className="placeholder-icon">🧾</div>
                <strong>Unable to preview image</strong>
                <span>Please choose a different receipt image.</span>
              </div>
            )}

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
          <button className="download-excel-button" onClick={downloadExcel}>
            <span className="download-icon">↓</span>
            <span>Download Excel</span>
          </button>

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
