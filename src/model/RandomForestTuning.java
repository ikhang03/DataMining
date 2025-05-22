package model;

import model.Command;
import preprocessing.dataImporter;
import weka.classifiers.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.RandomForest;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

import static preprocessing.dataImporter.validSource;

public class RandomForestTuning implements Command {
    public static void main(String[] args) {
        Command cmd = new RandomForestTuning();
        cmd.exec(dataImporter.trainSource, dataImporter.testSource);
    }

    private static void setClassIndex(Instances dataset) {
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1);
        }
    }

    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        try {
            Instances trainingDataSet = trainSource.getDataSet();

            Instances testingDataSet = testSource.getDataSet();

            Instances validDataset = validSource.getDataSet();

            setClassIndex(trainingDataSet);
            setClassIndex(testingDataSet);
            setClassIndex(validDataset);

            double bestAccuracy = -1;
            String[] bestOptions = null;

            RandomForest forest = new RandomForest();

            CVParameterSelection ps = new CVParameterSelection();
            ps.setClassifier(forest);
            ps.setNumFolds(10);
            ps.addCVParameter("I 10 30 3");
            ps.addCVParameter("K 0 5 1");

            ps.buildClassifier(validDataset);

            RandomForest tempRf = new RandomForest();
            tempRf.setOptions(ps.getBestClassifierOptions());
            tempRf.buildClassifier(validDataset);

            Evaluation eval = new Evaluation(validDataset);
            eval.crossValidateModel(tempRf, validDataset, 5, new java.util.Random(1));

            if (eval.pctCorrect() > bestAccuracy) {
                bestAccuracy = eval.pctCorrect();
                bestOptions = ps.getBestClassifierOptions();
            }

            System.out.println("\nPost-tuning RandomForest\n======\n");

            System.out.println("Best Parameters: " + String.join(" ", bestOptions));

            RandomForest finalRf = new RandomForest();
            finalRf.setOptions(bestOptions);
            finalRf.buildClassifier(trainingDataSet);

            Evaluation testEval = new Evaluation(trainingDataSet);
            testEval.evaluateModel(finalRf, testingDataSet);

            System.out.println(testEval.toSummaryString());

            System.out.println(testEval.toMatrixString("=== Confusion matrix ==="));

            System.out.println("Correct % = " + testEval.pctCorrect());
            System.out.println("Incorrect % = " + testEval.pctIncorrect());
            System.out.println("AUC = " + testEval.areaUnderROC(1));
            System.out.println("Kappa = " + testEval.kappa());
            System.out.println("MAE = " + testEval.meanAbsoluteError());
            System.out.println("RMSE = " + testEval.rootMeanSquaredError());
            System.out.println("RAE = " + testEval.relativeAbsoluteError());
            System.out.println("RRSE = " + testEval.rootRelativeSquaredError());
            System.out.println("Precision = " + testEval.precision(1));
            System.out.println("Recall = " + testEval.recall(1));
            System.out.println("F-Measure = " + testEval.fMeasure(1));
            System.out.println("Error Rate = " + testEval.errorRate());
            System.out.println(testEval.toClassDetailsString());
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
