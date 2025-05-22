package model;

import preprocessing.dataImporter;
import weka.classifiers.evaluation.Evaluation;
import weka.classifiers.rules.OneR;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.StringToNominal;

public class OneRClassifier implements Command {
    public static void main(String[] args) {
        // Fix: Create an instance of OneRClassifier, not LogisticRegressionClassifier
        Command cmd = new OneRClassifier();
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

            StringToNominal stringToNominal = new StringToNominal();
            stringToNominal.setAttributeRange("first-last"); // Convert all attributes
            stringToNominal.setInputFormat(trainDataset);
            trainDataset = Filter.useFilter(trainDataset, stringToNominal);
            testDataset = Filter.useFilter(testDataset, stringToNominal);

            OneR oner = new OneR();
            oner.setMinBucketSize(6);
            oner.buildClassifier(trainDataset);

            System.out.println("OneR classifier built successfully");
            System.out.println("OneR params: " + String.join(" ", oner.getOptions()));
            System.out.println("OneR model: \n" + oner);

            Evaluation eval = new Evaluation(trainDataset);
            eval.evaluateModel(oner, testDataset);

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
            System.out.println("Error in OneR classification:");
            e.printStackTrace();
        }
    }
}