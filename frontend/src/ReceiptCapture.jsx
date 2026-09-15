import { useEffect } from "react";
import { useReceiptScanner } from "./useReceiptScanner";

export default function ReceiptCapture({ onScanned }) {
  const { scanImage, isProcessing } = useReceiptScanner();

  useEffect(() => {
    if (!window.cv) {
      const cvScript = document.createElement("script");

      cvScript.src = "https://docs.opencv.org/4.x/opencv.js";

      cvScript.async = true;

      document.body.appendChild(cvScript);
    }

    if (!window.jscanify) {
      const jsScript = document.createElement("script");

      jsScript.src =
        "https://cdn.jsdelivr.net/npm/jscanify@1.4.3/src/jscanify.min.js";

      jsScript.async = true;

      document.body.appendChild(jsScript);
    }
  }, []);

  async function handleFileChange(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    event.target.value = "";

    try {
      const dataUrl = await scanImage(file);
      onScanned?.(dataUrl);
    } catch (error) {
      console.error("Receipt scanning failed:", error);
    }
  }

  return (
    <input
      type="file"
      accept="image/*"
      capture="environment"
      onChange={handleFileChange}
      disabled={isProcessing}
      hidden
    />
  );
}
