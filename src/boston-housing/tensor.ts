import { DatasetType } from "./constant";

export class Tensor {
  constructor() {}

  init(dataset: DatasetType) {
    const rawTrainFeatures = dataset.trainFeatures;
    const rawTestFeatures = dataset.testFeatures;

    const { dataMean, dataStd } = this.determineMeanAndStddev(rawTrainFeatures);

    this._trainFeatures = this.normalizeTensor(
      rawTrainFeatures,
      dataMean,
      dataStd
    );
    this._trainTarget = dataset.trainTarget;

    this._testFeatures = this.normalizeTensor(
      rawTestFeatures,
      dataMean,
      dataStd
    );
    this._testTarget = dataset.testTarget;
  }

  get trainFeatures() {
    return this._trainFeatures ?? [];
  }
  get trainTarget() {
    return this._trainTarget ?? [];
  }
  get testFeatures() {
    return this._testFeatures ?? [];
  }
  get testTarget() {
    return this._testTarget ?? [];
  }

  private _trainFeatures?: number[][];
  private _trainTarget?: number[][];
  private _testFeatures?: number[][];
  private _testTarget?: number[][];

  private determineMeanAndStddev(data: number[][]): {
    dataMean: number[];
    dataStd: number[];
  } {
    const numFeatures = data[0].length;
    const dataMean: number[] = new Array(numFeatures).fill(0);
    const dataStd: number[] = new Array(numFeatures).fill(0);

    for (let i = 0; i < data.length; i++) {
      for (let j = 0; j < numFeatures; j++) {
        dataMean[j] += data[i][j];
      }
    }

    for (let j = 0; j < numFeatures; j++) {
      dataMean[j] /= data.length;
    }

    for (let i = 0; i < data.length; i++) {
      for (let j = 0; j < numFeatures; j++) {
        dataStd[j] += Math.pow(data[i][j] - dataMean[j], 2);
      }
    }

    for (let j = 0; j < numFeatures; j++) {
      dataStd[j] = Math.sqrt(dataStd[j] / data.length);
    }

    return { dataMean, dataStd };
  }

  private normalizeTensor(
    tensor: number[][],
    dataMean: number[],
    dataStd: number[]
  ): number[][] {
    const normalizedTensor: number[][] = [];

    for (let i = 0; i < tensor.length; i++) {
      const normalizedRow: number[] = [];

      for (let j = 0; j < tensor[i].length; j++) {
        const normalizedValue = (tensor[i][j] - dataMean[j]) / dataStd[j];
        normalizedRow.push(normalizedValue);
      }

      normalizedTensor.push(normalizedRow);
    }

    return normalizedTensor;
  }
}
