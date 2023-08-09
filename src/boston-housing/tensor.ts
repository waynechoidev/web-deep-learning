import { DatasetType } from "./constant";

export class Tensor {
  private _trainFeatures: number[][] = [[]];
  private _trainTarget: number[][] = [[]];
  private _testFeatures: number[][] = [[]];
  private _testTarget: number[][] = [[]];

  private _weights: number[] = [];
  private _bias = 0;
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

    this._weights = new Array(this._numFeatures)
      .fill(0)
      .map(() => Math.random());
  }

  get trainFeatures() {
    return this._trainFeatures;
  }
  get trainTarget() {
    return this._trainTarget;
  }
  get testFeatures() {
    return this._testFeatures;
  }
  get testTarget() {
    return this._testTarget;
  }

  get baseline() {
    const avgPrice = this.mean(this._trainTarget);

    const diff = this.sub(this._testTarget, avgPrice);
    const squaredDiff = this.square(diff);
    const baseline = this.mean(squaredDiff);

    return baseline;
  }

  trainModel(
    model: (
      features: number[],
      weights: number[],
      bias: number,
      numFeatures: number
    ) => number,
    numEpochs: number,
    learningRate: number
  ): { weights: number[]; bias: number } {
    for (let epoch = 0; epoch < numEpochs; epoch++) {
      let totalLoss = 0;

      for (let i = 0; i < this._trainFeatures.length; i++) {
        const prediction = model(
          this._trainFeatures[i],
          this._weights,
          this._bias,
          this._numFeatures
        );
        const error = this._trainTarget[i][0] - prediction;

        for (let j = 0; j < this._numFeatures; j++) {
          this._weights[j] += learningRate * error * this._trainFeatures[i][j];
        }
        this._bias += learningRate * error;

        totalLoss += error ** 2;
      }

      const meanLoss = totalLoss / this._trainFeatures.length;
      console.log(`Epoch ${epoch + 1}, Mean Loss: ${meanLoss}`);
    }

    return { weights: this._weights, bias: this._bias };
  }

  linearRegressionModel(
    features: number[],
    weights: number[],
    bias: number,
    numFeatures: number
  ): number {
    let prediction = bias;
    for (let j = 0; j < numFeatures; j++) {
      prediction += weights[j] * features[j];
    }
    return prediction;
  }

  // Private methods
  private get _numFeatures() {
    return this._trainFeatures[0].length;
  }

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

  private mean(data: number[][]): number {
    const sum = data.flat().reduce((acc, val) => acc + val, 0);
    return sum / (data.length * data[0].length);
  }

  private sub(data: number[][], value: number): number[][] {
    return data.map((row) => row.map((val) => val - value));
  }

  private square(data: number[][]): number[][] {
    return data.map((row) => row.map((val) => val * val));
  }
}
