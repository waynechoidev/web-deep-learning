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

document.addEventListener(
  "DOMContentLoaded",
  async () => {
    await data.loadData();

    tensors_lr.init(data.dataset);
    tensors_mlp.init(data.dataset);

    await tensors_lr.trainModel();
    await tensors_mlp.trainModel();
    // tensors.testModel();
  },
  false
);
