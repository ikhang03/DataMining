package model;

import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.meta.CVParameterSelection;
import weka.classifiers.trees.J48;
import weka.core.Instances;

import static preprocessing.dataImporter.*;

public class J48Tuning {


    /**
     * Set the class index to the last attribute if it is not already set.
     *
     * @param dataset the Instances object
     */
    private static void setClassIndex(Instances dataset) {
        if (dataset.classIndex() == -1) {
            dataset.setClassIndex(dataset.numAttributes() - 1);
        }
    }

    public void exec() {
        try {
            Instances trainDataset = trainSource.getDataSet();
            Instances testDataset = testSource.getDataSet();
            Instances validDataset = validSource.getDataSet();

            setClassIndex(trainDataset);
            setClassIndex(testDataset);
            setClassIndex(validDataset);

            CVParameterSelection ps = new CVParameterSelection();
            ps.setClassifier(new J48());
            ps.setNumFolds(10);

            ps.addCVParameter("M 2 8 4");

            ps.buildClassifier(validDataset);

            System.out.println("Best Parameters: " + String.join(" ", ps.getBestClassifierOptions()));

            J48 j48 = new J48();
            j48.setOptions(ps.getBestClassifierOptions());
            j48.buildClassifier(trainDataset);

            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(j48, testDataset);

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

        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}