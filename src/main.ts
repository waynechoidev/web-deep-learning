import { BostonHousingDataset } from "./boston-housing/dataset";
import "./style.css";

document.querySelector<HTMLDivElement>("#app")!.innerHTML =
  "<div>Hello Deep Learning</div>";

const data = new BostonHousingDataset();

document.addEventListener(
  "DOMContentLoaded",
  async () => {
    await data.loadData();
    console.log(data.dataset);
  },
  false
);
