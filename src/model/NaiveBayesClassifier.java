package model;

import model.Command;
import preprocessing.dataImporter;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.evaluation.Evaluation;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class NaiveBayesClassifier implements Command {

    public static void main(String[] args) {
        Command cmd = new NaiveBayesClassifier();
        cmd.exec(dataImporter.trainSource, dataImporter.testSource);
    }

    @Override
    public void exec(DataSource trainSource, DataSource testSource) {
        try {
            Instances trainDataset = trainSource.getDataSet();

            Instances testDataset = testSource.getDataSet();

            if (trainDataset.classIndex() == -1) {
                trainDataset.setClassIndex(trainDataset.numAttributes() - 1);
            }

            if (testDataset.classIndex() == -1) {
                testDataset.setClassIndex(testDataset.numAttributes() - 1);
            }


            NaiveBayes nb = new NaiveBayes();
            nb.buildClassifier(trainDataset);

            System.out.println("NB params" + String.join(" ", nb.getOptions()));

            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(nb, testDataset);

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