let model;

async function loadImageAndLabel(imageUrl, labelArray) {
  return new Promise((resolve) => {
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imageUrl;
    img.onload = () => {
      const tensor = tf.browser
        .fromPixels(img)
        .resizeNearestNeighbor([128, 128])
        .toFloat()
        .div(tf.scalar(255.0))
        .expandDims(0); 
        // [1, 128, 128, 3]

        // used for unwanted label like nan to 0
      const percentValues = labelArray.map((entry) =>
        parseFloat(entry.percentage.replace("%", ""))
      );
      while (percentValues.length < 11) {
        percentValues.push(0);
      }

      resolve({ tensor, label: percentValues });
    };
  });
}

async function trainModel() {
  const imageFolder = "dataset/images/";
  const labelMap = await fetch("dataset/labels.json").then((res) => res.json());

  const imageTensors = [];
  const labelTensors = [];

  for (const [fileName, labelArray] of Object.entries(labelMap)) {
    const { tensor, label } = await loadImageAndLabel(
      imageFolder + fileName,
      labelArray
    );
    imageTensors.push(tensor);
    labelTensors.push(tf.tensor2d([label])); // [1, 11]
  }

  const xs = tf.concat(imageTensors); // [N, 128, 128, 3]
  const ys = tf.concat(labelTensors); // [N, 11]

  const modelLocal = tf.sequential();
  modelLocal.add(
    tf.layers.conv2d({
      inputShape: [128, 128, 3],
      filters: 16,
      kernelSize: 3,
      activation: "relu",
    })
  );
  modelLocal.add(tf.layers.maxPooling2d({ poolSize: 2 }));
  modelLocal.add(tf.layers.flatten());
  modelLocal.add(tf.layers.dense({ units: 64, activation: "relu" }));
  modelLocal.add(tf.layers.dense({ units: 11 }));

  modelLocal.compile({
    optimizer: tf.train.adam(),
    loss: "meanSquaredError",
    metrics: ["mae"],
  });

  await modelLocal.fit(xs, ys, {
    batchSize: 16,
    epochs: 20,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training", tab: "Training" },
      ["loss", "mae"],
      { height: 200 }
    ),
  });

  await modelLocal.save("downloads://my-model");
  model = modelLocal;
  console.log("Training done and model saved/downloaded.");
}

async function loadModel() {
  try {
    model = await tf.loadLayersModel("my-model/model.json");
    console.log("Model loaded!");
  } catch (err) {
    console.error("Failed to load model:", err);
  }
}

function preprocessImage(imgElement) {
  return tf.tidy(() => {
    return tf.browser
      .fromPixels(imgElement)
      .resizeBilinear([128, 128])
      .toFloat()
      .div(tf.scalar(255.0))
      .expandDims(0);
  });
}

document
  .getElementById("imageInput")
  .addEventListener("change", async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const img = new Image();
    const reader = new FileReader();

    reader.onload = (e) => {
      img.src = e.target.result;
    };
    reader.readAsDataURL(file);

    img.onload = async () => {
      const canvas = document.getElementById("preview");
      const ctx = canvas.getContext("2d");
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, 128, 128);

      if (!model) {
        alert("Model not loaded yet!");
        return;
      }

      const inputTensor = preprocessImage(img);
      const prediction = model.predict(inputTensor);
      const values = await prediction.data();

      document.getElementById("result").innerText =
        "Predicted values: " +
        Array.from(values)
          .map((v) => v.toFixed(2))
          .join(", ");

      inputTensor.dispose();
      prediction.dispose();
    };
  });

// On page load, either train the model or load an existing one
document.addEventListener("DOMContentLoaded", async () => {
  // Uncomment below line if you want to train model on page load
  // await trainModel();

  // Or load the saved model from folder
  await loadModel();
});
