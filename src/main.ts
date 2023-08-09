import { ModelType } from "./boston-housing/constant";
import { BostonHousingDataset } from "./boston-housing/dataset";
import { Tensor } from "./boston-housing/tensor";
import "./style.css";

document.querySelector<HTMLDivElement>("#app")!.innerHTML =
  "<div>Hello Deep Learning</div>";

const data = new BostonHousingDataset();
const tensors = new Tensor(ModelType.LinearRegressionModel);

document.addEventListener(
  "DOMContentLoaded",
  async () => {
    await data.loadData();
    tensors.init(data.dataset);

    console.log(data.dataset.trainFeatures);
    console.log(tensors.trainFeatures);
    console.log(tensors.baseline);
    const learningRate = 0.001;
    const numEpochs = 500;
    const linearRegressionModel = tensors.trainModel(numEpochs, learningRate);
    console.log(linearRegressionModel);
    tensors.testModel();
  },
  false
);
