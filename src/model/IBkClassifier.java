package model;

import model.Command;
import preprocessing.dataImporter;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.lazy.IBk;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class IBkClassifier implements Command {
    public static void main(String[] args) {
        Command cmd = new IBkClassifier();
        cmd.exec(dataImporter.trainSource, dataImporter.testSource);
    }

    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        // Start timing the overall execution
        long startTimeTotal = System.currentTimeMillis();

        try {
            Instances trainDataset = trainSource.getDataSet();
            Instances testDataset = testSource.getDataSet();

            if (trainDataset.classIndex() == -1) {
                trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
            }

            if (testDataset.classIndex() == -1) {
                testDataset.setClassIndex(testDataset.numAttributes() - 1);
            }

            IBk ibk = new IBk();

            // Start timing the training phase
            long startTimeTraining = System.currentTimeMillis();
            ibk.buildClassifier(trainDataset);
            long endTimeTraining = System.currentTimeMillis();
            long trainingTime = endTimeTraining - startTimeTraining;

            System.out.println("Classifier built successfully");
            System.out.println("IBk params" + String.join(" ", ibk.getOptions()));
            System.out.println("Starting evaluation...");

            Evaluation eval = new Evaluation(trainDataset);

            // Start timing the testing phase
            long startTimeTesting = System.currentTimeMillis();
            eval.evaluateModel(ibk, testDataset);
            long endTimeTesting = System.currentTimeMillis();
            long testingTime = endTimeTesting - startTimeTesting;

            System.out.println(eval.toSummaryString("\nResults\n======\n", false));

            System.out.println("Confusion Matrix:\n" + eval.toMatrixString());

            System.out.println("Correct % = " + eval.pctCorrect());
            System.out.println("Incorrect % = " + eval.pctIncorrect());
            System.out.println("AUC = " + eval.areaUnderROC(1));
            System.out.println("Kappa = " + eval.kappa());
            System.out.println("MAE = " + eval.meanAbsoluteError());
            System.out.println("RMSE = " + eval.rootMeanSquaredError());
            System.out.println("RAE = " + eval.relativeAbsoluteError());
            System.out.println("RRSE = " + eval.rootRelativeSquaredError());
            System.out.println("Precision = " + eval.precision(1));
            System.out.println("Recall = " + eval.recall(1));
            System.out.println("F-Measure = " + eval.fMeasure(1));
            System.out.println("Error Rate = " + eval.errorRate());
            System.out.println(eval.toClassDetailsString());

            // Calculate total execution time
            long endTimeTotal = System.currentTimeMillis();
            long totalTime = endTimeTotal - startTimeTotal;

            // Print timing information
            System.out.println("\n=== Runtime Information ===");
            System.out.println("Training Time: " + trainingTime + " ms");
            System.out.println("Testing Time: " + testingTime + " ms");
            System.out.println("Total Execution Time: " + totalTime + " ms");

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}