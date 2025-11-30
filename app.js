let session = null;
let scaler = null;

async function loadModel() {
  session = await ort.InferenceSession.create("./skincare_regression.onnx");

  const resp = await fetch("./skincare_scaler.json");
  scaler = await resp.json();
}

function getInputVector() {
  return [
    parseFloat(document.getElementById("hyal").value),
    parseFloat(document.getElementById("niac").value),
    parseFloat(document.getElementById("ret").value),
    parseFloat(document.getElementById("vitc").value),
    parseInt(document.getElementById("frag").value),
    parseFloat(document.getElementById("price").value)
  ];
}

function applyScaler(x) {
  let mean = scaler.mean;
  let scale = scaler.scale;
  return x.map((v, i) => (v - mean[i]) / scale[i]);
}

async function runModel() {
  if (!session) await loadModel();

  let x = getInputVector();
  let x_scaled = applyScaler(x);

  // Convert to tensor
  const tensor = new ort.Tensor("float32", Float32Array.from(x_scaled), [1, 6]);

  
  const output = await session.run({ input: tensor });
  const hydration = output.output.data[0];

  document.getElementById("result").innerText =
    `Prediction: ${hydration.toFixed(2)} / 100`;

  document.getElementById("debug").innerText =
    JSON.stringify({ features: x, scaled: x_scaled }, null, 2);
}

// Load model on startup
loadModel();
