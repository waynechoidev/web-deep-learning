import { Graph } from "./graph";
import { DatasetType, ModelType } from "./constant";

export class Tensor {
  private _modelType: ModelType;
  private _learningRate: number;
  private _numEpochs: number;
  private _graph: Graph;

  private _trainFeatures: number[][] = [[]];
  private _trainTarget: number[][] = [[]];
  private _testFeatures: number[][] = [[]];
  private _testTarget: number[][] = [[]];

  private _weights: number[] = [];
  private _hiddenActivations: number[] = [];
  private _bias = 0;

  constructor(
    modelType: ModelType,
    learningRate: number,
    numEpochs: number,
    graph: Graph
  ) {
    this._modelType = modelType;
    this._graph = graph;
    this._learningRate = learningRate;
    this._numEpochs = numEpochs;

    this._hiddenActivations = new Array(this.HIDDEN_UNITS).fill(0);
  }

  private HIDDEN_UNITS = 50 as const;

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

    this._weights = new Array(this._numFeatures + this.HIDDEN_UNITS)
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

  public async trainModel() {
    const model = this.getModel(this._modelType);

    let epoch = 0;
    const updateChart = () => {
      if (epoch++ < this._numEpochs) {
        const predictions = this.feedForward(
          model,
          this._trainFeatures,
          this._weights,
          this._bias
        );

        const meanLoss = this.calculateLoss(predictions);
        console.log(`Epoch ${epoch + 1}, Mean Loss: ${meanLoss}`);
        this._graph.update({ epoch, meanLoss });

        this.backPropagation(
          this._trainFeatures,
          this._trainTarget,
          predictions,
          this._learningRate
        );

        // Schedule the next update
        requestAnimationFrame(updateChart);
      }
    };
    // Start the initial update
    requestAnimationFrame(updateChart);
  }

  public async testModel() {
    const model = this.getModel(this._modelType);

    const testPredictions = this.feedForward(
      model,
      this._testFeatures,
      this._weights,
      this._bias
    );

    let totalLoss = 0; // Initialize total loss

    testPredictions.forEach((pred, index) => {
      const target = this._testTarget[index][0];
      const loss = Math.pow(target - pred, 2); // Calculate loss for each prediction
      totalLoss += loss; // Accumulate the losses
      // console.log(`Target: ${target}, Pred: ${pred}, Loss: ${loss}`);
    });

    const meanLoss = totalLoss / testPredictions.length; // Calculate mean loss
    console.log(`Mean Test Loss: ${meanLoss}`);
  }

  // Models
  private _linearRegressionModel = (
    features: number[],
    weights: number[],
    bias: number,
    numFeatures: number
  ): number => {
    let prediction = bias;
    for (let j = 0; j < numFeatures; j++) {
      prediction += weights[j] * features[j];
    }
    return prediction;
  };

  private _multiLayerPerceptronModel = (
    features: number[],
    weights: number[],
    bias: number,
    numFeatures: number
  ): number => {
    // Calculate hidden layer activations
    for (let i = 0; i < this.HIDDEN_UNITS; i++) {
      let activation = bias;
      for (let j = 0; j < numFeatures; j++) {
        activation += weights[j] * features[j];
      }
      this._hiddenActivations[i] = this.sigmoid(activation);
    }

    // Calculate output
    let output = 0;
    for (let i = 0; i < this.HIDDEN_UNITS; i++) {
      output += this._hiddenActivations[i] * weights[this._numFeatures + i];
    }

    return output;
  };

  // Private methods
  private get _numFeatures() {
    return this._trainFeatures[0].length;
  }

  private getModel(type: ModelType) {
    switch (type) {
      case ModelType.LinearRegressionModel:
        return this._linearRegressionModel;
      case ModelType.MultiLayerPerceptronModel:
        return this._multiLayerPerceptronModel;
    }
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

  private feedForward(
    model: any,
    features: number[][],
    weights: number[],
    bias: number
  ): number[] {
    const predictions: number[] = [];

    for (let i = 0; i < features.length; i++) {
      const prediction = model(features[i], weights, bias, this._numFeatures);
      predictions.push(prediction);
    }

    return predictions;
  }

  private calculateLoss(predictions: number[]): number {
    let totalLoss = 0;

    for (let i = 0; i < this._trainFeatures.length; i++) {
      const error = this._trainTarget[i][0] - predictions[i];
      totalLoss += error ** 2;
    }

    return totalLoss / this._trainFeatures.length;
  }

  private backPropagation(
    features: number[][],
    target: number[][],
    predictions: number[],
    learningRate: number
  ) {
    const newWeights = [...this._weights];
    let newBias = this._bias;

    for (let i = 0; i < features.length; i++) {
      const error = target[i][0] - predictions[i];

      for (let j = 0; j < this._numFeatures; j++) {
        newWeights[j] += learningRate * error * features[i][j];
      }
      newBias += learningRate * error;

      for (let k = 0; k < this.HIDDEN_UNITS; k++) {
        const hiddenError = error * newWeights[this._numFeatures + k];
        const delta = hiddenError * this._hiddenActivations[k];
        newWeights[this._numFeatures + k] += learningRate * delta;
      }
    }

    this._weights = newWeights;
    this._bias = newBias;
  }

  private sigmoid(x: number): number {
    return 1 / (1 + Math.exp(-x));
  }
}
