// Get the PDF container and canvas element
const pdfContainer = document.getElementById('pdf-container');
const pdfCanvas = document.getElementById('pdf-render');

// Load the PDF
pdfjsLib.getDocument('CATAM-10.15-Final.pdf').promise.then((pdf) => {
  // Fetch the first page
  return pdf.getPage(1);
}).then((page) => {
const viewport = page.getViewport({ scale: 1.5 });
pdfCanvas.width = viewport.width;
pdfCanvas.height = viewport.height;

// Render the PDF page into the canvas
const renderContext = {
    canvasContext: pdfCanvas.getContext('2d'),
    viewport: viewport,
};
page.render(renderContext);
});