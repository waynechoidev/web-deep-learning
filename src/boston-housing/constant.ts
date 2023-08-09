export const BASE_URL =
  "https://storage.googleapis.com/tfjs-examples/multivariate-linear-regression/data/";
export const TRAIN_FEATURES_FN = "train-data.csv";
export const TRAIN_TARGET_FN = "train-target.csv";
export const TEST_FEATURES_FN = "test-data.csv";
export const TEST_TARGET_FN = "test-target.csv";

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

export enum ModelType {
  LinearRegressionModel = 1,
  MultiLayerPerceptronModel = 2,
}
