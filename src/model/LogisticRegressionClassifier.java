package model;

import preprocessing.dataImporter;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.functions.Logistic;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.supervised.instance.SMOTE;
import weka.filters.unsupervised.attribute.Normalize;
import weka.filters.unsupervised.attribute.StringToNominal;

public class LogisticRegressionClassifier implements Command {
    public static void main(String[] args) {
        Command cmd = new LogisticRegressionClassifier();
        cmd.exec(dataImporter.trainSource, dataImporter.testSource);
    }

    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        // Start timing the overall execution
        long startTimeTotal = System.currentTimeMillis();

        try {
            // Start timing the data preprocessing phase
            long startTimePreprocessing = System.currentTimeMillis();

            System.out.println("Loading datasets...");
            Instances trainDataset = trainSource.getDataSet();
            Instances testDataset = testSource.getDataSet();

            if (trainDataset.classIndex() == -1) {
                trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
            }

            if (testDataset.classIndex() == -1) {
                testDataset.setClassIndex(testDataset.numAttributes() - 1);
            }

            System.out.println("Converting string attributes to nominal...");
            StringToNominal stringToNominal = new StringToNominal();
            stringToNominal.setInputFormat(trainDataset);
            stringToNominal.setOptions(new String[]{"-R", "first-last"});
            trainDataset = Filter.useFilter(trainDataset, stringToNominal);
            testDataset = Filter.useFilter(testDataset, stringToNominal);

            System.out.println("Normalizing numeric attributes...");
            Normalize normalize = new Normalize();
            normalize.setInputFormat(trainDataset);
            trainDataset = Filter.useFilter(trainDataset, normalize);
            testDataset = Filter.useFilter(testDataset, normalize);

            // Check if this is a binary classification problem and apply SMOTE if needed
            if (trainDataset.numClasses() == 2) {
                int[] classCounts = new int[trainDataset.numClasses()];
                for (int i = 0; i < trainDataset.numInstances(); i++) {
                    classCounts[(int) trainDataset.instance(i).classValue()]++;
                }

                double ratio = Math.min(classCounts[0], classCounts[1]) / (double) Math.max(classCounts[0], classCounts[1]);

                // Apply SMOTE if class imbalance is detected
                if (ratio < 0.8) {
                    try {
                        System.out.println("Class imbalance detected. Applying SMOTE...");
                        SMOTE smote = new SMOTE();
                        smote.setInputFormat(trainDataset);
                        trainDataset = Filter.useFilter(trainDataset, smote);
                        System.out.println("SMOTE applied successfully");
                    } catch (Exception e) {
                        System.out.println("SMOTE failed, continuing without balancing: " + e.getMessage());
                    }
                }
            } else {
                System.out.println("Skipping SMOTE as this is not a binary classification problem");
            }

            long endTimePreprocessing = System.currentTimeMillis();
            long preprocessingTime = endTimePreprocessing - startTimePreprocessing;

            // Create and configure Logistic Regression classifier
            Logistic lr = new Logistic();
            // Configure parameters
            lr.setRidge(0.5); // Regularization parameter
            lr.setMaxIts(100); // Maximum iterations

            // Build classifier
            System.out.println("Building Logistic Regression classifier...");

            // Start timing the training phase
            long startTimeTraining = System.currentTimeMillis();
            lr.buildClassifier(trainDataset);
            long endTimeTraining = System.currentTimeMillis();
            long trainingTime = endTimeTraining - startTimeTraining;

            System.out.println("LR parameters: " + String.join(" ", lr.getOptions()));

            // Evaluate model
            System.out.println("Evaluating Logistic Regression classifier...");
            Evaluation evaluation = new Evaluation(trainDataset);

            // Start timing the testing phase
            long startTimeTesting = System.currentTimeMillis();
            evaluation.evaluateModel(lr, testDataset);
            long endTimeTesting = System.currentTimeMillis();
            long testingTime = endTimeTesting - startTimeTesting;

            // Output evaluation results
            System.out.println(evaluation.toSummaryString("\nResults\n======\n", false));

            // Print confusion matrix and other metrics
            System.out.println("Confusion Matrix:\n" + evaluation.toMatrixString());
            System.out.println("Correct % = " + evaluation.pctCorrect());
            System.out.println("Incorrect % = " + evaluation.pctIncorrect());
            System.out.println("AUC = " + evaluation.areaUnderROC(1));
            System.out.println("Kappa = " + evaluation.kappa());
            System.out.println("MAE = " + evaluation.meanAbsoluteError());
            System.out.println("RMSE = " + evaluation.rootMeanSquaredError());
            System.out.println("RAE = " + evaluation.relativeAbsoluteError());
            System.out.println("RRSE = " + evaluation.rootRelativeSquaredError());
            System.out.println("Precision = " + evaluation.precision(1));
            System.out.println("Recall = " + evaluation.recall(1));
            System.out.println("F-Measure = " + evaluation.fMeasure(1));
            System.out.println("Error Rate = " + evaluation.errorRate());
            System.out.println(evaluation.toClassDetailsString());

            // Calculate total execution time
            long endTimeTotal = System.currentTimeMillis();
            long totalTime = endTimeTotal - startTimeTotal;

            // Print timing information
            System.out.println("\n=== Runtime Information ===");
            System.out.println("Preprocessing Time: " + preprocessingTime + " ms");
            System.out.println("Training Time: " + trainingTime + " ms");
            System.out.println("Testing Time: " + testingTime + " ms");
            System.out.println("Total Execution Time: " + totalTime + " ms");

        } catch (Exception e) {
            System.err.println("Error in Logistic Regression classification:");
            e.printStackTrace();
        }
    }
}