async function getData() {
  const carsDataResponse = await fetch(
    "https://storage.googleapis.com/tfjs-tutorials/carsData.json"
  );

  const data = await carsDataResponse.json();
  const cleaned = data
    .map((d) => ({
      mpg: d.Miles_per_Gallon,
      horsepower: d.Horsepower,
    }))
    .filter((f) => f.mpg !== null && f.horsepower != null);
  return cleaned;
}

function createModel() {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [1], units: 15, useBias: true }));
  model.add(tf.layers.dense({ units: 1, useBias: true }));
  return model;
}

function convertToTensor(data) {
  return tf.tidy(() => {
    tf.util.shuffle(data);

    const inputs = data.map((d) => d.horsepower);
    const labels = data.map((d) => d.mpg);

    if (inputs.length === 0 || labels.length === 0) {
      throw new Error("Inputs or labels are empty.");
    }

    const inputTensor = tf.tensor2d(inputs, [inputs.length, 1]);
    const labelTensor = tf.tensor2d(labels, [labels.length, 1]);

    const inputMax = inputTensor.max();
    const inputMin = inputTensor.min();
    const labelMax = labelTensor.max();
    const labelMin = labelTensor.min();

    const normalizedInputs = inputTensor
      .sub(inputMin)
      .div(inputMax.sub(inputMin));
    const normalizedLabels = labelTensor
      .sub(labelMin)
      .div(labelMax.sub(labelMin));

    return {
      inputs: normalizedInputs,
      labels: normalizedLabels,
      inputMax,
      inputMin,
      labelMax,
      labelMin,
    };
  });
}

async function trainModel(model, inputs, labels) {
  model.compile({
    optimizer: tf.train.adam(),
    loss: tf.losses.meanSquaredError,
    metrics: ["mse"],
  });

  const batchSize = 32;
  const epochs = 50;

  return await model.fit(inputs, labels, {
    batchSize,
    epochs,
    shuffle: true,
    callbacks: tfvis.show.fitCallbacks(
      { name: "Training Performance" },
      ["loss", "mse"],
      { height: 200, callbacks: ["onEpochEnd"] }
    ),
  });
}

function testModel(model, inputData, normalizationData) {
  const { inputMax, inputMin, labelMin, labelMax } = normalizationData;

  const [xs, preds] = tf.tidy(() => {
    const xsNorm = tf.linspace(0, 1, 100);
    const predictions = model.predict(xsNorm.reshape([100, 1]));

    const unNormXs = xsNorm.mul(inputMax.sub(inputMin)).add(inputMin);
    const unNormPreds = predictions.mul(labelMax.sub(labelMin)).add(labelMin);

    return [unNormXs.dataSync(), unNormPreds.dataSync()];
  });

  console.log(xs, preds);
  const predictedPoints = Array.from(xs).map((val, i) => {
    return { x: val, y: preds[i] };
  });
  const originalPoints = inputData.map((d) => ({
    x: d.horsepower,
    y: d.mpg,
  }));

  tfvis.render.scatterplot(
    { name: "Model Predictions vs Original Data" },
    {
      values: [originalPoints, predictedPoints],
      series: ["original", "predicted"],
    },
    {
      xLabel: "Horsepower",
      yLabel: "MPG",
      height: 300,
    }
  );
}

document.addEventListener("DOMContentLoaded", async () => {
  const model = createModel();
  const data = await getData();
  const tensorData = convertToTensor(data);
  const { inputs, labels } = tensorData;

  await trainModel(model, inputs, labels);
  testModel(model, data, tensorData);
  console.log("Done Training");
});
