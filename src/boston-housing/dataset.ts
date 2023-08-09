import * as Papa from "papaparse";

const BASE_URL =
  "https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/";
const TRAIN_FEATURES_FN = "train-data.csv";
const TRAIN_TARGET_FN = "train-target.csv";
const TEST_FEATURES_FN = "test-data.csv";
const TEST_TARGET_FN = "test-target.csv";

export const featureDescriptions = [
  "per capita crime rate by town", // CRIM
  "proportion of residential land zoned for lots over 25,000 sq.ft.", // ZN
  "proportion of non-retail business acres per town.", // INDUS
  "Charles River dummy variable (1 if tract bounds river; 0 otherwise)", // CHAS
  "nitric oxides concentration (parts per 10 million)", // NOX
  "average number of rooms per dwelling", // RM
  "proportion of owner-occupied units built prior to 1940", // AGE
  "weighted distances to five Boston employment centres", // DIS
  "index of accessibility to radial highways", // RAD
  "full-value property-tax rate per $10,000", // TAX
  "pupil-teacher ratio by town", // PTRATIO
  "% lower status of the population", // LSTAT
  "Median value of owner-occupied homes in $1000's", // MEDV
];
export type DatasetType = {
  trainFeatures: number[][];
  trainTarget: number[][];
  testFeatures: number[][];
  testTarget: number[][];
};

export class BostonHousingDataset {
  constructor() {}

  async loadData() {
    console.log(`----- start parsing -----`);
    [
      this._trainFeatures,
      this._trainTarget,
      this._testFeatures,
      this._testTarget,
    ] = await Promise.all([
      this.loadCsv(TRAIN_FEATURES_FN),
      this.loadCsv(TRAIN_TARGET_FN),
      this.loadCsv(TEST_FEATURES_FN),
      this.loadCsv(TEST_TARGET_FN),
    ]);

    this.shuffle(this._trainFeatures, this._trainTarget);
    this.shuffle(this._testFeatures, this._testTarget);
    console.log(`----- parsing has done -----`);
  }

  get numTrainFeatures() {
    return this._trainFeatures?.[0].length ?? 0;
  }

  get dataset(): DatasetType {
    return {
      trainFeatures: this._trainFeatures ?? [],
      trainTarget: this._trainTarget ?? [],
      testFeatures: this._testFeatures ?? [],
      testTarget: this._testTarget ?? [],
    };
  }

  private _trainFeatures?: number[][];
  private _trainTarget?: number[][];
  private _testFeatures?: number[][];
  private _testTarget?: number[][];

  private async loadCsv(filename: string) {
    return new Promise<number[][]>((resolve) => {
      const url = `${BASE_URL}${filename}`;
      Papa.parse<object>(url, {
        download: true,
        header: true,
        complete: (results) => {
          resolve(this.parseCsv(results["data"]));
        },
      });
    });
  }

  private async parseCsv(data: object[]) {
    return new Promise<number[][]>((resolve) => {
      const res = data.map((item) => {
        return Object.values(item).map((value) => parseFloat(value));
      });
      resolve(res);
    });
  }

  private async shuffle(data: number[][], target: number[][]) {
    let counter = data.length;
    while (counter--) {
      const index = (Math.random() * counter) | 0;

      const tempData = data[counter];
      data[counter] = data[index];
      data[index] = tempData;

      const tempTarget = target[counter];
      target[counter] = target[index];
      target[index] = tempTarget;
    }
  }
}
