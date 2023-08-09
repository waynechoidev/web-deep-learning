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
  },
  false
);
