import { ModelType } from "./constant";
import { BostonHousingDataset } from "./dataset";
import { Tensor } from "./tensor";
import "./style.css";
import { Graph } from "./graph";

// Dataset
const data = new BostonHousingDataset();

// Linear Regression Model
const graph_lr = new Graph("graph-lr");
const tensors_lr = new Tensor(
  ModelType.LinearRegressionModel,
  0.0001,
  200,
  graph_lr
);

// Multi Layer Perceptron Model
const graph_mlp = new Graph("graph-mlp");
const tensors_mlp = new Tensor(
  ModelType.MultiLayerPerceptronModel,
  0.0001,
  200,
  graph_mlp
);

document.getElementById("load-button")?.addEventListener("click", async () => {
  await data.loadData();

  tensors_lr.init(data.dataset);
  tensors_mlp.init(data.dataset);

  const baseline = tensors_lr.baseline;
  document.getElementById(
    "base-loss"
  )!.innerHTML = `Baseline Loss (Mean Squared Error): ${baseline.toFixed(2)}`;

  (document.getElementById("train-button-lr") as HTMLButtonElement).disabled =
    false;
  (document.getElementById("train-button-mlp") as HTMLButtonElement).disabled =
    false;
});

document
  .getElementById("train-button-lr")
  ?.addEventListener("click", () => tensors_lr.trainModel());

document
  .getElementById("test-button-lr")
  ?.addEventListener("click", () => tensors_lr.testModel());

document
  .getElementById("train-button-mlp")
  ?.addEventListener("click", () => tensors_mlp.trainModel());

document
  .getElementById("test-button-mlp")
  ?.addEventListener("click", () => tensors_mlp.testModel());
