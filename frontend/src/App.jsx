import { useState } from "react";

function App() {
  const [image, setImage] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  function handleImageChange(event) {
    const file = event.target.files[0];

    if (file) {
      setImage(file);
      setResult(null);
      setError(null);
    }
  }

  async function processReceipt() {
    if (!image) {
      return;
    }

    setLoading(true);
    setError(null);

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

  return (
    <main>
      <h1>Receipt OCR</h1>

      <input type="file" accept="image/*" onChange={handleImageChange} />

      {image && (
        <div>
          <p>Selected: {image.name}</p>

          <img
            src={URL.createObjectURL(image)}
            alt="Receipt preview"
            width="300"
          />

          <br />

          <button onClick={processReceipt} disabled={loading}>
            {loading ? "Processing..." : "Process Receipt"}
          </button>
        </div>
      )}

      {error && <p>{error}</p>}

      {result && (
        <div>
          <h2>Extracted Information</h2>

          <p>Company: {result.company}</p>
          <p>Date: {result.date}</p>
          <p>TIN: {result.tin}</p>
          <p>Invoice Number: {result.invoice_number}</p>
          <p>VATable Sales: {result.vatable_sales}</p>
          <p>VAT Amount: {result.vat_amount}</p>
          <p>Total: {result.total}</p>
          <p>VAT Valid: {result.vat_valid ? "Yes" : "No"}</p>
        </div>
      )}
    </main>
  );
}

export default App;
