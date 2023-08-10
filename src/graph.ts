import { Chart } from "chart.js/auto";
import { ChartDataType } from "./constant";

export class Graph {
  private _chart: Chart;
  private _labels: number[] = [];
  private _data: number[] = [];

  constructor() {
    this._chart = new Chart(
      document.getElementById("graph") as HTMLCanvasElement,
      {
        type: "line",
        data: {
          labels: this._labels,
          datasets: [
            {
              data: this._data,
              // backgroundColor: "rgba(255, 99, 132, 0.2)",
              // borderColor: "rgba(255, 99, 132, 1)",
              // borderWidth: 1,
            },
          ],
        },
        options: {
          responsive: true,
          plugins: {
            title: {
              display: true,
              text: "Linear",
            },
            legend: {
              display: false,
            },
          },
          elements: {
            point: {
              radius: 0,
            },
          },
          interaction: {
            intersect: true,
          },
          scales: {
            x: {
              display: true,
              title: {
                display: true,
                text: "Epoch",
              },
            },
            y: {
              display: true,
              title: {
                display: true,
                text: "Mean Loss",
              },
              suggestedMin: 0,
              suggestedMax: 600,
            },
          },
        },
      }
    );
  }

  update(data: ChartDataType) {
    this._labels.push(data.epoch);
    this._data.push(data.meanLoss);
    this._chart.update();
  }
}
