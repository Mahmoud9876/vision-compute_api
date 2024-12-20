const express = require('express');
const bodyParser = require('body-parser');
const multer = require('multer');
const ort = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

const app = express();
const PORT = 3000;

// Middleware
app.use(bodyParser.json());

// Configurer Multer pour gérer les fichiers téléchargés
const upload = multer({ dest: 'uploads/' });

// Charger le modèle YOLO (ONNX)
let session;
(async () => {
  try {
    console.log("Loading YOLO model...");
    const modelPath = path.join(__dirname, 'models', 'best.onnx'); // Remplacez par votre chemin
    session = await ort.InferenceSession.create(modelPath);
    console.log("YOLO model loaded successfully!");
  } catch (error) {
    console.error("Error loading YOLO model:", error);
  }
})();

// Fonction pour effectuer des prédictions
async function runYoloModel(imageBuffer) {
  const tensor = preprocessImage(imageBuffer);

  // Exécuter l'inférence
  const results = await session.run({ images: tensor });
  return postprocessResults(results);
}

// Prétraitement des images (conversion en tenseur)
function preprocessImage(imageBuffer) {
  const Jimp = require('jimp');

  return Jimp.read(imageBuffer).then((image) => {
    // Redimensionner l'image à la taille attendue par le modèle (640x640 pour YOLOv8)
    image.resize(640, 640);

    // Normaliser les pixels (0-255 -> 0-1)
    const data = Float32Array.from(image.bitmap.data).map((v) => v / 255.0);

    // Créer un tenseur
    return new ort.Tensor('float32', data, [1, 3, 640, 640]);
  });
}

// Post-traitement des résultats
function postprocessResults(results) {
  const detections = results.output0; // Remplacez par le nom de sortie réel de votre modèle
  const formattedDetections = detections.map((detection) => ({
    classId: detection[0], // Classe détectée
    confidence: detection[1], // Score de confiance
    box: {
      x: detection[2],
      y: detection[3],
      width: detection[4],
      height: detection[5],
    },
  }));

  return formattedDetections;
}

// Endpoint pour détecter des objets dans une image
app.post('/predict', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No image uploaded.' });
    }

    const imageBuffer = fs.readFileSync(req.file.path);

    // Effectuer une prédiction avec YOLO
    const predictions = await runYoloModel(imageBuffer);

    // Supprimer le fichier temporaire
    fs.unlinkSync(req.file.path);

    // Retourner les résultats
    res.json({ predictions });
  } catch (error) {
    console.error("Prediction error:", error);
    res.status(500).json({ error: error });
  }
});

// Lancer le serveur
app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
