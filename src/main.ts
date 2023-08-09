import { ModelType } from "./boston-housing/constant";
import { BostonHousingDataset } from "./boston-housing/dataset";
import { Tensor } from "./boston-housing/tensor";
import "./style.css";

document.querySelector<HTMLDivElement>("#app")!.innerHTML =
  "<div>Hello Deep Learning</div>";

const data = new BostonHousingDataset();
const tensors = new Tensor();

document.addEventListener(
  "DOMContentLoaded",
  async () => {
    await data.loadData();
    tensors.init(data.dataset);

    console.log(data.dataset.trainFeatures);
    console.log(tensors.trainFeatures);
    console.log(tensors.baseline);
    const learningRate = 0.001;
    const numEpochs = 200;
    const linearRegressionModel = tensors.trainModel(
      ModelType.LinearRegressionModel,
      numEpochs,
      learningRate
    );
    console.log(linearRegressionModel);
  },
  false
);
