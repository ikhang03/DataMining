package model;

import model.Command;
import preprocessing.dataImporter;
import weka.classifiers.Evaluation;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class RandomForestClassifier implements Command {

    public static void main(String[] args) {
        Command cmd = new RandomForestClassifier();
        cmd.exec(dataImporter.trainSource, dataImporter.testSource);
    }

    private static void setClassIndex(Instances dataset) {
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1);
        }
    }

    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        // Start timing the overall execution
        long startTimeTotal = System.currentTimeMillis();

        try {
            Instances trainingDataSet = trainSource.getDataSet();
            Instances testingDataSet = testSource.getDataSet();

            setClassIndex(trainingDataSet);
            setClassIndex(testingDataSet);

            RandomForest forest = new RandomForest();

            // Start timing the training phase
            long startTimeTraining = System.currentTimeMillis();
            forest.buildClassifier(trainingDataSet);
            long endTimeTraining = System.currentTimeMillis();
            long trainingTime = endTimeTraining - startTimeTraining;

            Evaluation eval = new Evaluation(trainingDataSet);

            // Start timing the testing phase
            long startTimeTesting = System.currentTimeMillis();
            eval.evaluateModel(forest, testingDataSet);
            long endTimeTesting = System.currentTimeMillis();
            long testingTime = endTimeTesting - startTimeTesting;

            System.out.println("RandomForest parameters: " + String.join(" ", forest.getOptions()));

            System.out.println(eval.toSummaryString("\nPre-tuning RandomForest\n======\n", false));

            System.out.println(eval.toMatrixString("=== Confusion matrix ==="));

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