import { ModelType } from "./constant";
import { BostonHousingDataset } from "./dataset";
import { Tensor } from "./tensor";
import "./style.css";
import { Graph } from "./graph";

const graph = new Graph();
const data = new BostonHousingDataset();
const tensors = new Tensor(ModelType.MultiLayerPerceptronModel, graph);

document.addEventListener(
  "DOMContentLoaded",
  async () => {
    await data.loadData();
    tensors.init(data.dataset);

    console.log(data.dataset.trainFeatures);
    console.log(tensors.trainFeatures);
    console.log(tensors.baseline);
    const learningRate = 0.0001;
    const numEpochs = 200;
    await tensors.trainModel(numEpochs, learningRate);
    tensors.testModel();
  },
  false
);
